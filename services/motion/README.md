# services/motion/ · 本地无界面 MediaPipe → VMC 捕捉服务

基于 Google 官方 **MediaPipe Tasks**（直接复用 3 个 `.task` 模型，总计 ~17 MB）+ 自适应 1€ 滤波的轻量级单摄像头捕捉子服务，输出标准 **VMC / OSC UDP** 数据包（默认 `127.0.0.1:39539`）：

- **52 通道标准 ARKit 表情 + 头部 3D 旋转**：`face_landmarker.task`
- **上半身（脊柱 / 胸口 / 双臂）**：`pose_landmarker_lite.task` / `pose_landmarker_full.task`
- **双手手腕 + 10 根手指（30 节指骨）**：`hand_landmarker.task`

## 日常使用
直接在 `AlicePet.app` 顶部菜单栏 **「动作与面部捕捉 (VMC)」** 一键开启/关闭；关闭时 `AlicePet` 直接终止子进程，摄像头与内存 100% 释放。

## 命令行手动调试 / 自测
```bash
cd services/motion
uv sync --python 3.12
.venv/bin/python tracker.py --self-test
.venv/bin/python tracker.py --port 39539 --camera 0 --body 1 --hands 1 --mirror 1
```
