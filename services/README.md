# services/ · 进程外 AI 子服务目录（预留）

为了保持 `AlicePet` 主程序极致轻量、省电且稳定，所有重型 AI 推理或外部桥接服务都将放在本目录中，作为**按需启停的独立子服务**运行（通过本地 Unix Socket / HTTP / WebSocket 与主程序通信）：

- `services/motion/`：动作生成与骨架重定向服务（如 **Kimodo / kimodo.cpp**、**ARDY**、VRMA 动作桥接）。
- `services/llm/`：在线 / 本地 LLM 对话、记忆与情绪指令输出服务。
- `services/tts/`：在线 / 本地 TTS 语音合成与实时 LipSync 口型权重流服务。

> 注意：模型权重、Python 虚拟环境（`.venv`）、Node 依赖与运行缓存均已被根目录 `.gitignore` 排除，切勿提交到 Git。
