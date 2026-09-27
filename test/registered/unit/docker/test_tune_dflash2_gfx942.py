import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "docker/rocm-mi308x-glm52-pd/scripts/tune_dflash2_gfx942.py"
)


def load_driver():
    spec = importlib.util.spec_from_file_location("tune_dflash2_gfx942", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_missing_configs_supports_bf16_and_a8w8_logs():
    driver = load_driver()
    log = "\n".join(
        [
            "unrelated startup line",
            "[aiter] shape is M:100, N:3072, K:4096 dtype='torch.bfloat16' "
            "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, "
            "not found tuned config in /data/aiter_configs/aiter_bf16_tuned_merged.csv",
            "[aiter] shape is M:101, N:3072, K:4096 dtype='torch.bfloat16' "
            "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, "
            "not found tuned config in /data/aiter_configs/aiter_bf16_tuned_merged.csv",
            "[aiter] shape is M:100, N:4096, K:1536, not found tuned config in "
            "/data/aiter_configs/aiter_a8w8_blockscale_merged.csv",
            "[aiter] shape is M:101, N:4096, K:1536, not found tuned config in "
            "/data/aiter_configs/aiter_a8w8_blockscale_merged.csv",
        ]
    )

    records = driver.parse_missing_configs(log)

    assert [(record.gemm_type, record.m, record.n, record.k) for record in records] == [
        ("bf16", 100, 3072, 4096),
        ("bf16", 101, 3072, 4096),
        ("a8w8", 100, 4096, 1536),
        ("a8w8", 101, 4096, 1536),
    ]
    assert records[0].dtype == "torch.bfloat16"
    assert records[0].otype == "torch.bfloat16"
    assert records[0].bias is False
    assert records[0].scale_ab is False
    assert records[0].bpreshuffle is False


def test_group_and_build_requests_use_unique_padded_m_buckets():
    driver = load_driver()
    records = driver.parse_missing_configs(
        "\n".join(
            [
                "[aiter] shape is M:100, N:3072, K:4096 dtype='torch.bfloat16' "
                "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, "
                "not found tuned config in /data/aiter_configs/aiter_bf16_tuned_merged.csv",
                "[aiter] shape is M:101, N:3072, K:4096 dtype='torch.bfloat16' "
                "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, "
                "not found tuned config in /data/aiter_configs/aiter_bf16_tuned_merged.csv",
                "[aiter] shape is M:100, N:4096, K:1536, not found tuned config in "
                "/data/aiter_configs/aiter_a8w8_blockscale_merged.csv",
            ]
        )
    )

    groups = driver.group_missing_configs(records)
    requests = driver.build_tuning_requests(
        groups,
        pad_m=lambda m, n, k, graph_launch: 112 if graph_launch == 0 else 128,
    )

    assert [(group.gemm_type, group.n, group.k, group.raw_m) for group in groups] == [
        ("a8w8", 4096, 1536, (100,)),
        ("bf16", 3072, 4096, (100, 101)),
    ]
    assert [(request.gemm_type, request.m, request.n, request.k) for request in requests] == [
        ("a8w8", 112, 4096, 1536),
        ("bf16", 112, 3072, 4096),
    ]
    fallback_requests = driver.build_tuning_requests(
        groups,
        pad_m=lambda m, n, k, graph_launch: 112 if graph_launch == 0 else 128,
        include_gl1=True,
    )
    assert [
        (request.gemm_type, request.m, request.n, request.k)
        for request in fallback_requests
    ] == [
        ("a8w8", 112, 4096, 1536),
        ("a8w8", 128, 4096, 1536),
        ("bf16", 112, 3072, 4096),
        ("bf16", 128, 3072, 4096),
    ]


def test_render_untuned_csvs_uses_tuner_schemas():
    driver = load_driver()
    records = driver.parse_missing_configs(
        "\n".join(
            [
                "[aiter] shape is M:100, N:3072, K:4096 dtype='torch.bfloat16' "
                "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, "
                "not found tuned config in /data/aiter_configs/aiter_bf16_tuned_merged.csv",
                "[aiter] shape is M:100, N:4096, K:1536, not found tuned config in "
                "/data/aiter_configs/aiter_a8w8_blockscale_merged.csv",
            ]
        )
    )
    requests = driver.build_tuning_requests(
        driver.group_missing_configs(records),
        pad_m=lambda m, n, k, graph_launch: 112 if graph_launch == 0 else 128,
    )

    bf16_csv, a8w8_csv = driver.render_untuned_csvs(
        requests, gfx="gfx942", cu_num=80
    )

    assert bf16_csv == (
        "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n"
        "gfx942,80,112,3072,4096,False,torch.bfloat16,torch.bfloat16,False,False\n"
    )
    assert a8w8_csv == (
        "M,N,K\n112,4096,1536\n"
    )


def test_filter_tuned_rows_keeps_only_bounded_gfx942_kernels():
    driver = load_driver()
    tuned_csv = "\n".join(
        [
            "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw",
            "gfx942,80,112,3072,4096,False,torch.bfloat16,torch.bfloat16,False,False,torch,1,1,10.0,valid_kernel,0.01,1,1",
            "gfx950,80,112,3072,4096,False,torch.bfloat16,torch.bfloat16,False,False,torch,1,1,9.0,wrong_gfx,0.01,1,1",
            "gfx942,80,128,3072,4096,False,torch.bfloat16,torch.bfloat16,False,False,torch,1,1,9.0,None,0.01,1,1",
            "gfx942,80,160,3072,4096,False,torch.bfloat16,torch.bfloat16,False,False,torch,1,1,9.0,high_error,0.06,1,1",
        ]
    )

    filtered = driver.filter_tuned_rows(
        tuned_csv, gemm_type="bf16", gfx="gfx942", cu_num=80, max_err_ratio=0.05
    )

    assert filtered == (
        "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw\n"
        "gfx942,80,112,3072,4096,False,torch.bfloat16,torch.bfloat16,False,False,torch,1,1,10.0,valid_kernel,0.01,1,1\n"
    )


def test_filter_tuned_rows_supports_a8w8_schema():
    driver = load_driver()
    tuned_csv = "\n".join(
        [
            "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio",
            "gfx942,80,112,4096,1536,ck,8,2,10.0,valid_kernel,1,1,0.01",
            "gfx942,80,128,4096,1536,ck,8,2,9.0,None,1,1,0.01",
        ]
    )

    filtered = driver.filter_tuned_rows(
        tuned_csv, gemm_type="a8w8", gfx="gfx942", cu_num=80, max_err_ratio=0.05
    )

    assert filtered == (
        "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n"
        "gfx942,80,112,4096,1536,ck,8,2,10.0,valid_kernel,1,1,0.01\n"
    )


def test_prepare_files_writes_inputs_and_summary(tmp_path):
    driver = load_driver()
    log_path = tmp_path / "prefill.log"
    log_path.write_text(
        "[aiter] shape is M:100, N:3072, K:4096 dtype='torch.bfloat16' "
        "otype='torch.bfloat16' bias=False, scaleAB=False, bpreshuffle=False, "
        "not found tuned config in /data/aiter_configs/aiter_bf16_tuned_merged.csv\n"
        "[aiter] shape is M:100, N:4096, K:1536, not found tuned config in "
        "/data/aiter_configs/aiter_a8w8_blockscale_merged.csv\n",
        encoding="utf-8",
    )

    prepared = driver.prepare_files(
        log_path=log_path,
        output_dir=tmp_path,
        gfx="gfx942",
        cu_num=80,
        pad_m=lambda m, n, k, graph_launch: 112 if graph_launch == 0 else 128,
    )

    assert prepared.bf16_input.read_text(encoding="utf-8").splitlines()[0] == (
        "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle"
    )
    assert prepared.a8w8_input.read_text(encoding="utf-8").splitlines() == [
        "M,N,K",
        "112,4096,1536",
    ]
    assert prepared.summary["missing_lines"] == 2
    assert prepared.summary["bf16_shapes"] == 1
    assert prepared.summary["a8w8_shapes"] == 1


def test_build_tuner_command_enables_splitk_without_preshuffle(tmp_path):
    driver = load_driver()

    command = driver.build_tuner_command(
        gemm_type="a8w8",
        python_executable="/usr/bin/python3",
        tuner_path=Path("/sgl-workspace/aiter/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py"),
        input_path=Path("/tmp/input.csv"),
        output_path=Path("/tmp/output.csv"),
        mp=1,
        warmup=5,
        iters=101,
        err_ratio=0.05,
        timeout=1800,
        bf16_libtypes="torch,skinny,triton,opus",
        a8w8_libtype="both",
    )

    assert command == [
        "/usr/bin/python3",
        "/sgl-workspace/aiter/csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py",
        "-i",
        "/tmp/input.csv",
        "-o",
        "/tmp/output.csv",
        "--mp",
        "1",
        "--warmup",
        "5",
        "--iters",
        "101",
        "--errRatio",
        "0.05",
        "--timeout",
        "1800",
        "--shape_grouped",
        "--splitK",
        "--libtype",
        "both",
    ]
    assert "--preshuffle" not in command


def test_aiter_padded_m_matches_upstream_bucket_rules():
    driver = load_driver()

    assert [
        driver.aiter_padded_m(raw_m, n=3072, k=4096, graph_launch=0)
        for raw_m in (68, 100, 257, 1025, 4097, 7838)
    ] == [80, 112, 288, 1088, 4224, 7936]
    assert [
        driver.aiter_padded_m(raw_m, n=3072, k=4096, graph_launch=1)
        for raw_m in (68, 7838, 8193)
    ] == [128, 8192, 16384]
    assert driver.aiter_padded_m(8193, n=4097, k=4096, graph_launch=1) == 8192
