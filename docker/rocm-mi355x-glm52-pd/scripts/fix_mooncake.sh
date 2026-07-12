#!/usr/bin/env bash
# Quick-fix: rebuild Mooncake at the known-working commit (01d1eb2a, the version
# sglang v0.5.15 ships) inside the existing image, then docker commit to overwrite.
# main HEAD has a regression (SIGSEGV during PD warmup); this pins mooncake back.
set -eux

echo "[fix] rebuilding mooncake at 01d1eb2a inside image..."
docker rm -f mc_fix 2>/dev/null || true
docker run --name mc_fix sglang-glm52-mi355x-pd:latest bash -c '
  set -eux
  rm -rf /sgl-workspace/Mooncake
  git clone https://github.com/kvcache-ai/Mooncake.git /sgl-workspace/Mooncake
  cd /sgl-workspace/Mooncake
  git checkout 01d1eb2a7ec37fd5e20a88573e9b4956e7846e9a
  git submodule update --init --recursive
  bash dependencies.sh -y
  mkdir -p build && cd build
  cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON -DENABLE_MULTI_PROTOCOL=ON -DWITH_STORE=ON -DBUILD_UNIT_TESTS=OFF
  make -j$(nproc) && make install
  ldconfig
  python3 -c "import mooncake.engine as e; print(\"mooncake.engine OK:\", e.__file__)"
'
docker commit mc_fix sglang-glm52-mi355x-pd:latest
docker rm mc_fix
echo "[fix] done. image now has mooncake 01d1eb2a:"
docker images sglang-glm52-mi355x-pd:latest --format "{{.Repository}}:{{.Tag}} {{.Size}} {{.CreatedSince}}"
