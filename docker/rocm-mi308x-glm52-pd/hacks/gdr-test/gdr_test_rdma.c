/* Test GPU-direct RDMA using rdmacm (handles RoCE v2 QP setup automatically).
 * Server: ./gdr_test_rdma server <port>
 * Client: ./gdr_test_rdma client <server_ip> <port>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <rdma/rsocket.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
#include <dlfcn.h>

#define CHECK(cond, msg) do { if(cond) { fprintf(stderr, "FAIL %s: %s (errno=%d)\n", msg, strerror(errno), errno); exit(1); } } while(0)

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
    CHECK(!p_hipMalloc || !p_hipFree || !p_hipMemcpy || !p_hipMemset, "dlsym hip");
}
#define HIPCHECK(cmd) do { int e = (cmd); if(e != 0) { fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, p_hipGetErrorString ? p_hipGetErrorString(e) : "unknown"); exit(1); } } while(0)

#define BUF_SIZE 4096

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s server <port>\n", argv[0]);
        fprintf(stderr, "Usage: %s client <server_ip> <port>\n", argv[0]);
        return 1;
    }
    int is_server = (strcmp(argv[1], "server") == 0);
    int port;
    const char *server_ip = NULL;

    load_hip();

    struct rdma_cm_id *id = NULL;
    struct rdma_event_channel *ec = rdma_create_event_channel();
    CHECK(!ec, "rdma_create_event_channel");

    struct rdma_addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_port_space = RDMA_PS_TCP;

    char port_str[16];
    struct ibv_pd *pd = NULL;
    struct ibv_mr *mr = NULL;
    struct ibv_cq *cq = NULL;
    void *gpu_buf = NULL;

    if (is_server) {
        port = atoi(argv[2]);
        snprintf(port_str, sizeof(port_str), "%d", port);
        hints.ai_flags = RAI_PASSIVE;
        CHECK(rdma_getaddrinfo(NULL, port_str, &hints, &res), "rdma_getaddrinfo server");
        CHECK(rdma_create_id(ec, &id, NULL, RDMA_PS_TCP), "rdma_create_id server");
        CHECK(rdma_bind_addr(id, res->ai_src_addr), "rdma_bind_addr");
        CHECK(rdma_listen(id, 1), "rdma_listen");
        printf("Server listening on port %d...\n", port);

        struct rdma_cm_event *event;
        CHECK(rdma_get_cm_event(ec, &event), "rdma_get_cm_event connect");
        if (event->event != RDMA_CM_EVENT_CONNECT_REQUEST) {
            fprintf(stderr, "Unexpected event: %d\n", event->event);
            return 1;
        }
        struct rdma_cm_id *new_id = event->id;
        rdma_ack_cm_event(event);

        /* Use the new connection id */
        pd = ibv_alloc_pd(new_id->verbs);
        CHECK(!pd, "ibv_alloc_pd");
        cq = ibv_create_cq(new_id->verbs, 16, NULL, NULL, 0);
        CHECK(!cq, "ibv_create_cq");

        /* Allocate GPU memory */
        HIPCHECK(p_hipSetDevice(0));
        HIPCHECK(p_hipMalloc(&gpu_buf, BUF_SIZE));
        HIPCHECK(p_hipMemset(gpu_buf, 0, BUF_SIZE));
        printf("Server: GPU buffer allocated: ptr=%p size=%d (cleared, expecting 0x1234)\n", gpu_buf, BUF_SIZE);

        /* Register GPU memory with RDMA */
        mr = ibv_reg_mr(pd, gpu_buf, BUF_SIZE,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        CHECK(!mr, "ibv_reg_mr GPU server");
        printf("Server: ibv_reg_mr GPU SUCCESS: rkey=%u lkey=%u addr=%p\n", mr->rkey, mr->lkey, mr->addr);

        /* Accept connection */
        struct rdma_conn_param conn_param;
        memset(&conn_param, 0, sizeof(conn_param));
        conn_param.initiator_depth = 1;
        conn_param.responder_resources = 1;
        conn_param.rnr_retry_count = 7;

        /* Build QP */
        struct ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.send_cq = cq;
        qp_attr.recv_cq = cq;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr = 16;
        qp_attr.cap.max_recv_wr = 16;
        qp_attr.cap.max_send_sge = 4;
        qp_attr.cap.max_recv_sge = 4;

        CHECK(rdma_create_qp(new_id, pd, &qp_attr), "rdma_create_qp server");
        /* The QP is now at new_id->qp */
        CHECK(rdma_accept(new_id, &conn_param), "rdma_accept");

        /* Wait for established */
        CHECK(rdma_get_cm_event(ec, &event), "rdma_get_cm_event established");
        if (event->event != RDMA_CM_EVENT_ESTABLISHED) {
            fprintf(stderr, "Unexpected event: %d\n", event->event);
            return 1;
        }
        rdma_ack_cm_event(event);
        printf("Server: connection established!\n");

        /* Wait for client to do RDMA write */
        printf("Server: connection established, exchanging MR info...\n");

        /* Exchange MR info via RDMA send/recv (same as client) */
        void *ctrl_buf = malloc(256);
        struct ibv_mr *ctrl_mr = ibv_reg_mr(pd, ctrl_buf, 256, IBV_ACCESS_LOCAL_WRITE);
        CHECK(!ctrl_mr, "ibv_reg_mr ctrl server");

        struct { uint32_t rkey; uint64_t raddr; } my_mr, peer_mr;
        my_mr.rkey = mr->rkey;
        my_mr.raddr = (uint64_t)(uintptr_t)gpu_buf;

        /* Post recv first */
        struct ibv_sge ctrl_recv_sge;
        memset(&ctrl_recv_sge, 0, sizeof(ctrl_recv_sge));
        ctrl_recv_sge.addr = (uint64_t)(uintptr_t)ctrl_buf;
        ctrl_recv_sge.length = sizeof(peer_mr);
        ctrl_recv_sge.lkey = ctrl_mr->lkey;
        struct ibv_recv_wr recv_wr;
        memset(&recv_wr, 0, sizeof(recv_wr));
        recv_wr.sg_list = &ctrl_recv_sge;
        recv_wr.num_sge = 1;
        struct ibv_recv_wr *bad_recv_wr;
        CHECK(ibv_post_recv(new_id->qp, &recv_wr, &bad_recv_wr), "ibv_post_recv server");

        /* Post send with our MR info */
        memcpy(ctrl_buf + 128, &my_mr, sizeof(my_mr));
        struct ibv_sge ctrl_send_sge;
        memset(&ctrl_send_sge, 0, sizeof(ctrl_send_sge));
        ctrl_send_sge.addr = (uint64_t)(uintptr_t)(ctrl_buf + 128);
        ctrl_send_sge.length = sizeof(my_mr);
        ctrl_send_sge.lkey = ctrl_mr->lkey;
        struct ibv_send_wr send_wr;
        memset(&send_wr, 0, sizeof(send_wr));
        send_wr.sg_list = &ctrl_send_sge;
        send_wr.num_sge = 1;
        send_wr.opcode = IBV_WR_SEND;
        send_wr.send_flags = IBV_SEND_SIGNALED;
        struct ibv_send_wr *bad_send_wr;
        CHECK(ibv_post_send(new_id->qp, &send_wr, &bad_send_wr), "ibv_post_send server");

        /* Wait for completions */
        struct ibv_wc wc[2];
        int done = 0;
        while (done < 2) {
            int ne = ibv_poll_cq(cq, 2, wc);
            for (int i = 0; i < ne; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "Server WC error: %s (status=%d)\n", ibv_wc_status_str(wc[i].status), wc[i].status);
                    return 1;
                }
                done++;
            }
        }
        memcpy(&peer_mr, ctrl_buf, sizeof(peer_mr));
        printf("Server: peer client MR: rkey=%u raddr=0x%llx\n", peer_mr.rkey, (unsigned long long)peer_mr.raddr);

        printf("Server: waiting for RDMA write from client...\n");
        sleep(3);

        /* Check if GPU buffer received 0x1234 */
        uint32_t *host_check = (uint32_t*)malloc(BUF_SIZE);
        HIPCHECK(p_hipMemcpy(host_check, gpu_buf, BUF_SIZE, hipMemcpyDeviceToHost));
        int ok = 1, mismatches = 0;
        for (int i = 0; i < BUF_SIZE / 4; i++) {
            if (host_check[i] != 0x1234) {
                ok = 0;
                if (mismatches < 5) printf("Mismatch at offset %d: got 0x%x expected 0x1234\n", i*4, host_check[i]);
                mismatches++;
            }
        }
        free(host_check);
        if (ok) {
            printf("\n*** GPU-DIRECT RDMA TEST PASSED! ***\n");
            printf("Server GPU buffer received 0x1234 via RDMA write.\n");
            printf("GPU-direct RDMA WORKS on bnxt_re + MI308X!\n");
        } else {
            printf("\n*** GPU-DIRECT RDMA TEST FAILED! ***\n");
            printf("Data mismatch (%d errors).\n", mismatches);
            printf("ibv_reg_mr succeeded but RDMA write to GPU memory didn't work.\n");
            printf("This means GPU-direct RDMA registration works but transfers need kernel P2P support.\n");
        }

        /* Disconnect */
        rdma_disconnect(new_id);
        rdma_destroy_qp(new_id);
        ibv_dereg_mr(ctrl_mr);
        ibv_dereg_mr(mr);
        ibv_destroy_cq(cq);
        ibv_dealloc_pd(pd);
        free(ctrl_buf);
        p_hipFree(gpu_buf);
        rdma_destroy_id(new_id);

    } else {
        /* Client */
        server_ip = argv[2];
        port = atoi(argv[3]);
        snprintf(port_str, sizeof(port_str), "%d", port);
        CHECK(rdma_getaddrinfo((char*)server_ip, port_str, &hints, &res), "rdma_getaddrinfo client");
        CHECK(rdma_create_id(ec, &id, NULL, RDMA_PS_TCP), "rdma_create_id client");
        CHECK(rdma_resolve_addr(id, NULL, res->ai_dst_addr, 5000), "rdma_resolve_addr");

        struct rdma_cm_event *event;
        CHECK(rdma_get_cm_event(ec, &event), "rdma_get_cm_event addr_resolved");
        if (event->event != RDMA_CM_EVENT_ADDR_RESOLVED) { fprintf(stderr, "Unexpected: %d\n", event->event); return 1; }
        rdma_ack_cm_event(event);

        CHECK(rdma_resolve_route(id, 5000), "rdma_resolve_route");
        CHECK(rdma_get_cm_event(ec, &event), "rdma_get_cm_event route_resolved");
        if (event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) { fprintf(stderr, "Unexpected: %d\n", event->event); return 1; }
        rdma_ack_cm_event(event);

        /* Now id->verbs is valid */
        pd = ibv_alloc_pd(id->verbs);
        CHECK(!pd, "ibv_alloc_pd client");
        cq = ibv_create_cq(id->verbs, 16, NULL, NULL, 0);
        CHECK(!cq, "ibv_create_cq client");

        /* Allocate GPU memory and fill with 0x1234 */
        HIPCHECK(p_hipSetDevice(0));
        HIPCHECK(p_hipMalloc(&gpu_buf, BUF_SIZE));
        uint32_t *host_tmp = (uint32_t*)malloc(BUF_SIZE);
        for (int i = 0; i < BUF_SIZE / 4; i++) host_tmp[i] = 0x1234;
        HIPCHECK(p_hipMemcpy(gpu_buf, host_tmp, BUF_SIZE, hipMemcpyHostToDevice));
        free(host_tmp);
        printf("Client: GPU buffer allocated: ptr=%p size=%d (filled with 0x1234)\n", gpu_buf, BUF_SIZE);

        /* Register GPU memory */
        mr = ibv_reg_mr(pd, gpu_buf, BUF_SIZE,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        CHECK(!mr, "ibv_reg_mr GPU client");
        printf("Client: ibv_reg_mr GPU SUCCESS: rkey=%u lkey=%u addr=%p\n", mr->rkey, mr->lkey, mr->addr);

        /* Build QP */
        struct ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.send_cq = cq;
        qp_attr.recv_cq = cq;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr = 16;
        qp_attr.cap.max_recv_wr = 16;
        qp_attr.cap.max_send_sge = 4;
        qp_attr.cap.max_recv_sge = 4;
        CHECK(rdma_create_qp(id, pd, &qp_attr), "rdma_create_qp client");

        /* Connect */
        struct rdma_conn_param conn_param;
        memset(&conn_param, 0, sizeof(conn_param));
        conn_param.initiator_depth = 1;
        conn_param.responder_resources = 1;
        conn_param.rnr_retry_count = 7;
        CHECK(rdma_connect(id, &conn_param), "rdma_connect");
        CHECK(rdma_get_cm_event(ec, &event), "rdma_get_cm_event connect");
        if (event->event != RDMA_CM_EVENT_ESTABLISHED) { fprintf(stderr, "Unexpected: %d\n", event->event); return 1; }

        /* Get server's rkey and raddr from private data or exchange via TCP */
        /* rdmacm established - now we need the server's MR info. Since we can't
         * easily exchange private data with rdmacm, let's use a simple approach:
         * the server's rkey/raddr are sent as rdmacm private data. But for simplicity,
         * let's exchange via a separate TCP connection. */
        /* Actually, rdmacm private data is sent in connect/accept. Let's use that. */
        /* For now, let's exchange via the established RDMA connection itself using send/recv */

        printf("Client: connection established! QP num=%u\n", id->qp->qp_num);

        /* Exchange MR info via RDMA send/recv */
        struct { uint32_t rkey; uint64_t raddr; } my_mr, peer_mr;
        my_mr.rkey = mr->rkey;
        my_mr.raddr = (uint64_t)(uintptr_t)gpu_buf;

        /* Post receive first */
        struct ibv_sge recv_sge;
        memset(&recv_sge, 0, sizeof(recv_sge));
        recv_sge.addr = (uint64_t)(uintptr_t)&peer_mr;
        recv_sge.length = sizeof(peer_mr);
        recv_sge.lkey = 0; /* stack buffer - need a registered MR for recv */
        /* Actually we need to register the recv buffer too. Let's use the GPU buf's MR
         * but that's GPU memory. Let's allocate a small host MR for control messages. */
        void *ctrl_buf = malloc(256);
        struct ibv_mr *ctrl_mr = ibv_reg_mr(pd, ctrl_buf, 256, IBV_ACCESS_LOCAL_WRITE);
        CHECK(!ctrl_mr, "ibv_reg_mr ctrl");

        /* Post recv */
        struct ibv_sge ctrl_recv_sge;
        memset(&ctrl_recv_sge, 0, sizeof(ctrl_recv_sge));
        ctrl_recv_sge.addr = (uint64_t)(uintptr_t)ctrl_buf;
        ctrl_recv_sge.length = sizeof(peer_mr);
        ctrl_recv_sge.lkey = ctrl_mr->lkey;
        struct ibv_recv_wr recv_wr;
        memset(&recv_wr, 0, sizeof(recv_wr));
        recv_wr.sg_list = &ctrl_recv_sge;
        recv_wr.num_sge = 1;
        struct ibv_recv_wr *bad_recv_wr;
        CHECK(ibv_post_recv(id->qp, &recv_wr, &bad_recv_wr), "ibv_post_recv");

        /* Post send with our MR info */
        memcpy(ctrl_buf + 128, &my_mr, sizeof(my_mr));  /* use second half for send */
        struct ibv_sge ctrl_send_sge;
        memset(&ctrl_send_sge, 0, sizeof(ctrl_send_sge));
        ctrl_send_sge.addr = (uint64_t)(uintptr_t)(ctrl_buf + 128);
        ctrl_send_sge.length = sizeof(my_mr);
        ctrl_send_sge.lkey = ctrl_mr->lkey;
        struct ibv_send_wr send_wr;
        memset(&send_wr, 0, sizeof(send_wr));
        send_wr.sg_list = &ctrl_send_sge;
        send_wr.num_sge = 1;
        send_wr.opcode = IBV_WR_SEND;
        send_wr.send_flags = IBV_SEND_SIGNALED;
        struct ibv_send_wr *bad_send_wr;
        CHECK(ibv_post_send(id->qp, &send_wr, &bad_send_wr), "ibv_post_send");

        /* Wait for completions */
        struct ibv_wc wc[2];
        int done = 0;
        while (done < 2) {
            int ne = ibv_poll_cq(cq, 2, wc);
            for (int i = 0; i < ne; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "WC error: %s (status=%d)\n", ibv_wc_status_str(wc[i].status), wc[i].status);
                    return 1;
                }
                done++;
            }
        }
        /* Extract peer MR info from recv buffer */
        memcpy(&peer_mr, ctrl_buf, sizeof(peer_mr));
        printf("Client: peer server MR: rkey=%u raddr=0x%llx\n", peer_mr.rkey, (unsigned long long)peer_mr.raddr);

        /* Now do RDMA write to server's GPU memory */
        struct ibv_sge write_sge;
        memset(&write_sge, 0, sizeof(write_sge));
        write_sge.addr = (uint64_t)(uintptr_t)gpu_buf;
        write_sge.length = BUF_SIZE;
        write_sge.lkey = mr->lkey;
        struct ibv_send_wr write_wr;
        memset(&write_wr, 0, sizeof(write_wr));
        write_wr.wr_id = 0xdead;
        write_wr.sg_list = &write_sge;
        write_wr.num_sge = 1;
        write_wr.opcode = IBV_WR_RDMA_WRITE;
        write_wr.send_flags = IBV_SEND_SIGNALED;
        write_wr.wr.rdma.remote_addr = peer_mr.raddr;
        write_wr.wr.rdma.rkey = peer_mr.rkey;

        printf("Client: issuing RDMA write (size=%d) to server GPU 0x%llx...\n", BUF_SIZE, (unsigned long long)peer_mr.raddr);
        CHECK(ibv_post_send(id->qp, &write_wr, &bad_send_wr), "ibv_post_send RDMA write");

        /* Wait for completion */
        struct ibv_wc write_wc;
        while (ibv_poll_cq(cq, 1, &write_wc) == 0);
        if (write_wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "FAILED: RDMA write WC error: %s (status=%d vendor_err=%u)\n",
                    ibv_wc_status_str(write_wc.status), write_wc.status, write_wc.vendor_err);
            fprintf(stderr, "==> GPU-direct RDMA transfer FAILED!\n");
            return 3;
        }
        printf("SUCCESS: RDMA write to server GPU completed!\n");

        rdma_ack_cm_event(event);
        sleep(1);
        rdma_disconnect(id);
        ibv_dereg_mr(ctrl_mr);
        ibv_dereg_mr(mr);
        ibv_destroy_cq(cq);
        ibv_dealloc_pd(pd);
        free(ctrl_buf);
        p_hipFree(gpu_buf);
        rdma_destroy_qp(id);
    }

    rdma_destroy_id(id);
    rdma_destroy_event_channel(ec);
    return 0;
}
