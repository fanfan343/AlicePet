# AlicePet

macOS（Apple Silicon）原生的超轻量 3D VRM 桌面伴侣，基于 **Swift + Metal + `VRMMetalKit`** 渲染，采用「原生极轻身体 + 进程外按需 AI 子服务 + ZeroClaw 大脑与 MCP 桥」解耦架构。

> 架构设计见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · 详细路线图见 [docs/ROADMAP.md](docs/ROADMAP.md) · 协作规则见 [AGENTS.md](AGENTS.md)
> 实时进度看板：[GitHub Projects v2](https://github.com/users/fanfan343/projects/1) · [Milestones](https://github.com/fanfan343/AlicePet/milestones) · [Issues](https://github.com/fanfan343/AlicePet/issues)

---

## 进度看板（Progress Dashboard）

### 1. 四种产品形态

| 形态 | 名称 | 状态 | 说明 |
| :--- | :--- | :--- | :--- |
| **Mode 1** | 普通桌宠模式 | ✅ 已上线 | 透明置顶陪伴，支持鼠标旋转/平移/缩放视角、换装、表情与物理 |
| **Mode 2** | 画中模式（Picture-World） | ✅ 核心完成 | 2D 插画/照片（及**空透明房间**）+ MoGe-2 硬件深度遮挡 + A* 避障 + VLM 自主识座落座；用户仅缩放/中键观察，角色移动由 AI 自主驱动 |
| **Mode 3** | Apple 原生对话 UI | 🗓️ 预留规划 | 原生独立聊天窗口（待定，先聚焦核心体验） |
| **Mode 4** | 纯状态栏极简模式 | 🗓️ 预留规划 | 仅保留顶部状态栏图标与小工具面板（Todo / 剪贴板 / 额度 / 系统状态） |

### 2. 阶段里程碑总览

| 阶段 | 里程碑主题 | 状态 | 核心交付 |
| :--- | :--- | :--- | :--- |
| **Stage 1–4** | 身体渲染、三层动画混合器与画中世界 | ✅ 100% | Swift+Metal 4× MSAA、三层 `AnimationMixer`、ARDY 20fps + 2-Bone IK、A2F-3D Claire、MoGe-2 深度轮廓贴边遮挡、A* 导航走去坐下 |
| **Stage 5** | Harness 大脑底座与「虚拟生命」闭环 | ✅ 100% | `LocalControlServer`（`Network.framework`）、纯 stdlib MCP 桥、`ZeroClaw` 底座、两段式 Markdown 记忆、8 轴连续情绪惯性状态机 |
| **Stage 6** | 表现力深化与语音闭环 | 🚧 进行中 (3/5) | ✅ 标准 `.vrma` 播放 + Layer 1 说话手势节拍与情绪体态<br>✅ 画中模式空透明房间（0 场景显存）<br>✅ VLM 零预设锚点自主识座落座验收<br>⬜ 本地语音通话闭环（STT + Smart Turn + TTS → A2F-3D）<br>⬜ 动态帧率省电策略与多 VRM 通用适配 |
| **Stage 7+** | 远期形态与创意池 | 💡 Backlog | Mode 3 原生对话窗、Mode 4 状态栏小工具、XR Animator 画中互动、Folia 风格后台音乐联动动画 |

---

## 目录结构

- `AlicePet/`：SwiftPM macOS 原生主应用源码（Metal 渲染、三层混合器、画中世界、情绪状态机、控制桥）与打包脚本
- `services/`：完全解耦的进程外子服务（关闭即 0 VRAM）
  - `services/bridge/`：零依赖 Python stdlib MCP 桥（`alicepet_mcp.py`）与 ZeroClaw 配置模板
  - `services/scene/`：MoGe-2 深度 + RGB 边缘细化 + 地面拟合 + 可走网格生成
  - `services/ardy/`：ARDY Mini 20fps 文本生成动作（VMC/OSC）
  - `services/tts/`：Audio2Face-3D v2.3.1 Claire（10 维情绪 + 52 ARKit 口型）与 LAM
  - `services/motion/`：本机 MediaPipe 上半身/面部/十指捕捉
- `soul/`：两段式 Markdown 记忆模板（`SOUL.md` / `PROFILE.md` / `MEMORIES.md` / `TODO.md`，`episodes/` 已忽略）
- `tools/`：资产提取工具与一键视觉回归脚本（`verify_visuals.sh`）
- `docs/`：架构文档（[ARCHITECTURE.md](docs/ARCHITECTURE.md)）与路线图（[ROADMAP.md](docs/ROADMAP.md)）

## 快速开始

```bash
# 1. 构建并签名 Release 版 AlicePet.app（推荐日常运行，低 CPU / 省电）
./AlicePet/build_app.sh

# 2. 启动桌面伴侣
open ./AlicePet.app

# 3. 跑一键视觉回归截图（输出 snapshots/latest/contact_sheet.png）
./tools/verify_visuals.sh
```

运行日志：`~/Library/Logs/AlicePet/alicepet.log`（含 `[Control]` / `[MCP]` / `[Emotion]` / `[ARDY]` / `[Perf]` 统一前缀，亦可通过菜单栏「打开日志文件夹」查看）。
