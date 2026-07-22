import json

before_text = (
    "> ⚠️ **更新（2026-07-21）**：本文档 2026-07-12 的根因结论（"
    "\"OFED 缺 peer_memory\"）已被**部分纠正**。新测试证实 amdgpu peermem 已正常加载，"
    "loopback GDR 可用，跨节点失败的真正根因是 **Mooncake fallback 分支缺 HipDeviceGuard "
    "+ worker 线程缺 hipSetDevice**。详见 "
    "[MI308X GDR 验证与 Mooncake 代码根因分析（2026-07-21）](/p/4026859496)。\n\n---\n\n"
)

args = {
    "id": 4025463879,
    "title": "MI308X 跨机 PD 分离 RDMA 排查记录 — GPU 显存 RDMA 根因确诊",
    "before": before_text,
}
with open("/tmp/iwiki_update_args.json", "w") as f:
    json.dump(args, f, ensure_ascii=False)
print("JSON written, before length:", len(before_text))
