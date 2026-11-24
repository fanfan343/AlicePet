# services/tts/ · 音频驱动 52 通道 ARKit 表情服务（LAM_Audio2Expression ONNX）

基于阿里通义 3DAIGC **`LAM_Audio2Expression`** 的 INT8 量化 ONNX 版本（`enconvo/avatar-lipsync-1-onnx` / `lam-int8.onnx`，约 146 MB，Apache 2.0 协议）：

- 输入：任意语音音频（自动解码为 16kHz 单声道）
- 推理：`onnxruntime` CPU 实时推理（Apple Silicon 上约 **20× 实时**，10 秒音频约半秒算完）
- 输出：**60fps 插值同步的 52 通道标准 ARKit BlendShapes + 轻度说话点头微动作**，经由本机 VMC (`127.0.0.1:39539`) 驱动 `AlicePet`，同时由 macOS `afplay` 严格对齐时钟播放声音。

## 在 AlicePet 中测试
打开 `AlicePet.app` 顶部菜单栏 **「测试功能 (Lab)」**：
- **「播放 10s VTuber 语音对口型测试 (LAM 52通道)」**
- **「选择任意音频文件对口型测试…」**

## 命令行单独调试
```bash
cd services/tts
uv sync --python 3.12
.venv/bin/python audio2face_vmc.py --self-test
.venv/bin/python audio2face_vmc.py --audio samples/vtuber_10s.wav --port 39539
```
