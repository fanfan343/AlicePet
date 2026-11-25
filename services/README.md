# services/ · 进程外 AI 子服务目录（预留）

为了保持 `AlicePet` 主程序极致轻量、省电且稳定，所有重型 AI 推理或外部桥接服务都将放在本目录中，作为**按需启停的独立子服务**运行（通过本地 Unix Socket / HTTP / WebSocket 与主程序通信）：

- `services/motion/`：本地无界面 MediaPipe 精捕服务（52 通道 ARKit 表情 + 上半身 + 十指，走本机 VMC/OSC 推送）。
- `services/ardy/`：ARDY Mini 文本生成全身动作服务（MiniLM 蒸馏文本编码器 + ARDY-Core40 纯 ONNX，20fps 自回归流式推 VMC，走独立混合通道）。
- `services/scene/`：画中世界（形态 B）单图米制深度与地平面预处理服务（微软 `MoGe-2` ViT-S，Apple Silicon MPS 约 1.6s 单图重建 `bg.png + depth.bin + scene.json`，跑完即退 0 常驻占用）。
- `services/llm/`：在线 / 本地 LLM 对话、记忆与情绪指令输出服务。
- `services/tts/`：音频驱动 52 通道 ARKit 表情与叠加头部微动服务，支持 **NVIDIA Audio2Face-3D v2.3.1 Claire**（ONNX + 52-D 缓存求解器，支持 10 种原生显式情绪与真实上脸眉眼动态）与 **`LAM_Audio2Expression`**（INT8 ONNX 极速纯口型兜底），走独立 `audioFaceChannel` 通道。

> 注意：模型权重、Python 虚拟环境（`.venv`）、Node 依赖与运行缓存均已被根目录 `.gitignore` 排除，切勿提交到 Git。
