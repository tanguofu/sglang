/* Test GPU-direct RDMA: register GPU memory via ibv_reg_mr and do RDMA write.
 * Uses ctypes-style HIP loading to avoid HIP header complications.
 * Usage on server: ./gdr_test server <dev_name> <port>
 * Usage on client: ./gdr_test client <dev_name> <server_ip> <port>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <infiniband/verbs.h>
#include <dlfcn.h>

#define CHECK(cond, msg) do { if(cond) { perror(msg); exit(1); } } while(0)

/* Load HIP dynamically */
typedef int (*hipMalloc_t)(void**, size_t);
typedef int (*hipFree_t)(void*);
typedef int (*hipMemcpy_t)(void*, const void*, size_t, int);
typedef int (*hipMemset_t)(void*, int, size_t);
typedef int (*hipSetDevice_t)(int);
typedef const char* (*hipGetErrorString_t)(int);

#define hipMemcpyHostToDevice 1
#define hipMemcpyDeviceToHost 2

static hipMalloc_t p_hipMalloc;
static hipFree_t p_hipFree;
static hipMemcpy_t p_hipMemcpy;
static hipMemset_t p_hipMemset;
static hipSetDevice_t p_hipSetDevice;
static hipGetErrorString_t p_hipGetErrorString;

static void load_hip() {
    void *h = dlopen("libamdhip64.so", RTLD_NOW);
    CHECK(!h, "dlopen libamdhip64.so");
    p_hipMalloc = (hipMalloc_t)dlsym(h, "hipMalloc");
    p_hipFree = (hipFree_t)dlsym(h, "hipFree");
    p_hipMemcpy = (hipMemcpy_t)dlsym(h, "hipMemcpy");
    p_hipMemset = (hipMemset_t)dlsym(h, "hipMemset");
    p_hipSetDevice = (hipSetDevice_t)dlsym(h, "hipSetDevice");
    p_hipGetErrorString = (hipGetErrorString_t)dlsym(h, "hipGetErrorString");
    CHECK(!p_hipMalloc || !p_hipFree || !p_hipMemcpy || !p_hipMemset || !p_hipSetDevice, "dlsym hip");
}
#define HIPCHECK(cmd) do { int e = (cmd); if(e != 0) { fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, p_hipGetErrorString ? p_hipGetErrorString(e) : "unknown"); exit(1); } } while(0)

struct Conn {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    void *buf;
    size_t size;
};

static void setup_qp(struct Conn *c) {
    c->cq = ibv_create_cq(c->ctx, 16, NULL, NULL, 0);
    CHECK(!c->cq, "ibv_create_cq");
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.send_cq = c->cq;
    attr.recv_cq = c->cq;
    attr.qp_type = IBV_QPT_RC;
    attr.cap.max_send_wr = 16;
    attr.cap.max_recv_wr = 16;
    attr.cap.max_send_sge = 4;
    attr.cap.max_recv_sge = 4;
    c->qp = ibv_create_qp(c->pd, &attr);
    CHECK(!c->qp, "ibv_create_qp");
}

static void modify_qp_to_init(struct Conn *c) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    CHECK(ibv_modify_qp(c->qp, &attr, IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS), "modify_qp_init");
}

static void modify_qp_to_rtr(struct Conn *c, uint32_t remote_qpn, int gid_index) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.traffic_class = 0;
    attr.ah_attr.dlid = 0;
    CHECK(ibv_modify_qp(c->qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER), "modify_qp_rtr");
}

static void modify_qp_to_rts(struct Conn *c) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    CHECK(ibv_modify_qp(c->qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC), "modify_qp_rts");
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <server|client> <dev_name> <port> [server_ip]\n", argv[0]);
        return 1;
    }
    int is_server = (strcmp(argv[1], "server") == 0);
    const char *dev_name = argv[2];
    const char *server_ip = NULL;
    int port = 0;
    if (is_server) {
        port = atoi(argv[3]);
    } else {
        server_ip = argv[3];
        port = atoi(argv[4]);
    }
    size_t buf_size = 4096;

    load_hip();

    /* Find RDMA device */
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    CHECK(!dev_list, "ibv_get_device_list");
    struct ibv_context *ctx = NULL;
    for (int i = 0; dev_list[i]; i++) {
        if (strcmp(ibv_get_device_name(dev_list[i]), dev_name) == 0) {
            ctx = ibv_open_device(dev_list[i]);
            break;
        }
    }
    CHECK(!ctx, "ibv_open_device");
    printf("Opened device: %s\n", dev_name);

    struct Conn c;
    memset(&c, 0, sizeof(c));
    c.ctx = ctx;
    c.pd = ibv_alloc_pd(ctx);
    CHECK(!c.pd, "ibv_alloc_pd");
    c.size = buf_size;

    /* Allocate GPU memory */
    HIPCHECK(p_hipSetDevice(0));
    HIPCHECK(p_hipMalloc(&c.buf, buf_size));
    printf("GPU buffer allocated: ptr=%p size=%zu\n", c.buf, buf_size);

    /* Initialize GPU buffer */
    if (is_server) {
        HIPCHECK(p_hipMemset(c.buf, 0, buf_size));
        printf("Server: buffer cleared (expecting to receive 0x1234 from client)\n");
    } else {
        uint32_t *host_tmp = (uint32_t*)malloc(buf_size);
        for (size_t i = 0; i < buf_size / 4; i++) host_tmp[i] = 0x1234;
        HIPCHECK(p_hipMemcpy(c.buf, host_tmp, buf_size, hipMemcpyHostToDevice));
        free(host_tmp);
        printf("Client: buffer filled with 0x1234\n");
    }

    /* Register GPU memory with RDMA — THIS IS THE KEY TEST */
    c.mr = ibv_reg_mr(c.pd, c.buf, buf_size,
                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!c.mr) {
        fprintf(stderr, "FAILED: ibv_reg_mr for GPU memory: %s (errno=%d)\n", strerror(errno), errno);
        fprintf(stderr, "==> GPU-direct RDMA NOT supported\n");
        return 2;
    }
    printf("SUCCESS: ibv_reg_mr for GPU memory: rkey=%u lkey=%u addr=%p\n", c.mr->rkey, c.mr->lkey, c.mr->addr);

    /* Setup QP */
    setup_qp(&c);
    modify_qp_to_init(&c);

    /* TCP socket for exchange */
    int sock;
    if (is_server) {
        int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1; setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = INADDR_ANY;
        CHECK(bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0, "bind");
        CHECK(listen(listen_fd, 1) < 0, "listen");
        printf("Server listening on port %d...\n", port);
        sock = accept(listen_fd, NULL, NULL);
        CHECK(sock < 0, "accept");
        close(listen_fd);
    } else {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, server_ip, &addr.sin_addr);
        printf("Client connecting to %s:%d...\n", server_ip, port);
        CHECK(connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0, "connect");
    }

    /* Exchange QP info via socket */
    struct __attribute__((packed)) { uint32_t qpn, rkey; uint64_t raddr; } my_info, peer_info;
    my_info.qpn = c.qp->qp_num;
    my_info.rkey = c.mr->rkey;
    my_info.raddr = (uint64_t)(uintptr_t)c.buf;
    CHECK(send(sock, &my_info, sizeof(my_info), 0) != (ssize_t)sizeof(my_info), "send info");
    CHECK(recv(sock, &peer_info, sizeof(peer_info), 0) != (ssize_t)sizeof(peer_info), "recv info");
    printf("Peer: qpn=%u rkey=%u raddr=0x%llx\n", peer_info.qpn, peer_info.rkey, (unsigned long long)peer_info.raddr);

    /* Modify QP to RTR and RTS — use GID index 3 (RoCE v2 IPv4) */
    modify_qp_to_rtr(&c, peer_info.qpn, 3);
    modify_qp_to_rts(&c);
    printf("QP ready (RTR->RTS)\n");

    if (!is_server) {
        /* Client: RDMA write 0x1234 pattern to server's GPU memory */
        struct ibv_sge sge;
        memset(&sge, 0, sizeof(sge));
        sge.addr = (uint64_t)(uintptr_t)c.buf;
        sge.length = (uint32_t)buf_size;
        sge.lkey = c.mr->lkey;

        struct ibv_send_wr wr;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id = 1;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = peer_info.raddr;
        wr.wr.rdma.rkey = peer_info.rkey;

        struct ibv_send_wr *bad_wr;
        printf("Client: issuing RDMA write (size=%zu) to remote GPU 0x%llx...\n", buf_size, (unsigned long long)peer_info.raddr);
        CHECK(ibv_post_send(c.qp, &wr, &bad_wr), "ibv_post_send");

        struct ibv_wc wc;
        int ne;
        do { ne = ibv_poll_cq(c.cq, 1, &wc); } while (ne == 0);
        CHECK(ne < 0, "poll_cq");
        if (wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "FAILED: RDMA write completed with error: %s (status=%d vendor_err=%u)\n",
                    ibv_wc_status_str(wc.status), wc.status, wc.vendor_err);
            fprintf(stderr, "==> GPU-direct RDMA transfer FAILED!\n");
            return 3;
        }
        printf("SUCCESS: RDMA write completed! (wr_id=%llu)\n", (unsigned long long)wc.wr_id);
    }

    /* Wait for transfer to complete */
    sleep(2);

    if (is_server) {
        /* Server: check if GPU buffer received 0x1234 */
        uint32_t *host_check = (uint32_t*)malloc(buf_size);
        HIPCHECK(p_hipMemcpy(host_check, c.buf, buf_size, hipMemcpyDeviceToHost));
        int ok = 1;
        int mismatches = 0;
        for (size_t i = 0; i < buf_size / 4; i++) {
            if (host_check[i] != 0x1234) {
                ok = 0;
                if (mismatches < 5) printf("Mismatch at offset %zu: got 0x%x expected 0x1234\n", i*4, host_check[i]);
                mismatches++;
            }
        }
        free(host_check);
        if (ok) {
            printf("\n*** GPU-DIRECT RDMA TEST PASSED! ***\n");
            printf("Server GPU buffer received 0x1234 pattern from client via RDMA write.\n");
            printf("GPU-direct RDMA (ibv_reg_mr on GPU memory + RDMA write) WORKS!\n");
        } else {
            printf("\n*** GPU-DIRECT RDMA TEST FAILED! ***\n");
            printf("Data mismatch (%d errors) — RDMA write did not reach GPU memory correctly.\n", mismatches);
            return 4;
        }
    } else {
        printf("\nClient done. Check server output for results.\n");
    }

    close(sock);
    ibv_dereg_mr(c.mr);
    ibv_destroy_qp(c.qp);
    ibv_destroy_cq(c.cq);
    ibv_dealloc_pd(c.pd);
    ibv_close_device(c.ctx);
    ibv_free_device_list(dev_list);
    p_hipFree(c.buf);
    return 0;
}
