# DeepSeek-V4.1-Flash on MI308X (gfx942): official ROCm build plus this tree.
#
# The base image installs sglang in editable mode from /sgl-workspace/sglang/python,
# so replacing that package replaces exactly the code the server imports -- no
# PYTHONPATH shim and no runtime patching.
#
# Plain COPY merges into the existing directory, which leaves modules that
# upstream renamed or deleted behind. The tree is cleared first and rebuilt from
# source, restoring only the prebuilt artefacts the base image ships (47 MB of
# AOT kernels and the Rust multimodal extension have no counterpart in git).
#
# The entrypoint compiles the per-batch DSV4.1 kernels before opening readiness,
# because TileLang rebuilds its DSA prefill kernel per batch shape (~70 s on
# gfx942) and the artifact does not survive the process in this image.
#
# Build context is the parent directory of python/sglang, laid out as ./sglang.
FROM mirrors.tencent.com/ti-platform/sglang-glm52-308x:ds41-official-1007

RUN set -eux; \
    mv /sgl-workspace/sglang/python/sglang/kernels/aot/build /tmp/keep-aot; \
    mkdir -p /tmp/keep-rust; \
    cp /sgl-workspace/sglang/python/sglang/srt/rust_extensions/*.so /tmp/keep-rust/ 2>/dev/null || true; \
    rm -rf /sgl-workspace/sglang/python/sglang

COPY sglang /sgl-workspace/sglang/python/sglang
COPY warmup_ds41.py /usr/local/bin/warmup_ds41.py
COPY ds41_entrypoint.sh /usr/local/bin/ds41_entrypoint.sh

RUN set -eux; \
    mkdir -p /sgl-workspace/sglang/python/sglang/kernels/aot; \
    mv /tmp/keep-aot /sgl-workspace/sglang/python/sglang/kernels/aot/build; \
    cp /tmp/keep-rust/*.so /sgl-workspace/sglang/python/sglang/srt/rust_extensions/; \
    rm -rf /tmp/keep-aot /tmp/keep-rust; \
    chmod +x /usr/local/bin/ds41_entrypoint.sh; \
    find /sgl-workspace/sglang/python/sglang -name '._*' -delete; \
    find /sgl-workspace/sglang/python/sglang -name '.DS_Store' -delete; \
    find /sgl-workspace/sglang/python/sglang -name '__pycache__' -type d -prune -exec rm -rf {} +; \
    find /sgl-workspace/sglang/python/sglang -name '*.pyc' -delete; \
    find /sgl-workspace/sglang/python/sglang -name '*darwin*.so' -delete

ENV PYTHONPATH=/sgl-workspace/sglang/python \
    SGLANG_HIP_DSV41_INDEX_K_FP8=1
LABEL gpu-arch=gfx942 maintainer=guofutan ds41-index-k=fp8

ENTRYPOINT ["/usr/local/bin/ds41_entrypoint.sh"]
