# services/ardy/ · ARDY Mini 文本驱动全身动作实时生成服务

基于 **`intsuc/Llama-3-ARDY-Mini-Core40-Browser`**（NVIDIA `ARDY-Core-RP-20FPS-Horizon40` 无损动作扩散模型 + `all-MiniLM-L6-v2` 蒸馏文本编码器，总计 ~652 MB 纯 ONNX）：

- **20 FPS 实时自回归流式生成**（60 FPS SLERP 插值推流至本机 VMC `127.0.0.1:39539`）
- **实时切换英文提示词**：动作不断档、基于前 40 帧历史无缝衔接
- **Core27 → VRM 1.0 重定向**：自动折叠 `Spine1/2/3 → Chest`，兼容无 `upperChest` 的 VRM 角色

## 命令行自测
```bash
cd services/ardy
uv sync --python 3.12
.venv/bin/python ardy_vmc.py --self-test --prompt "a person waves cheerfully with their right hand"
```
