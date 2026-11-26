# AlicePet 路线图（ROADMAP）

原则：**原生极轻、大脑与身体解耦、先验主线闭环再逐项打磨。** 详细架构见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

## 四种形态规划
- **Mode 1 · 普通桌宠模式**：透明置顶陪伴，用户可交互视角（已实现）
- **Mode 2 · 画中模式**：2D 插画/照片（及空透明房间）+ 硬件深度遮挡 + A* 避障，用户仅缩放/中键观察，动作由 LLM 自主驱动（已实现核心，持续补齐空房间与 VLM 锚点）
- **Mode 3 · 原生对话 UI**：Apple 原生独立对话窗（预留位置，暂不实现）
- **Mode 4 · 纯状态栏模式**：无窗口仅顶部状态栏图标，后续可挂 Todo/剪贴板/额度/系统状态小工具（预留位置）

---

## Stage 1–4：身体、渲染、动捕与画中世界（已完成主线）
- [x] macOS 原生 Swift + Metal + `VRMMetalKit` 4× MSAA 渲染、透明桌宠、换装/发型、独立道具与视觉回归闭环
- [x] 三层动画混合器 `AnimationMixer`（原版 Unity 肌肉动画 + 0.35s SLERP + Layer 1 头部微动 + Layer 2 下脸/上脸/视线三轨解耦）
- [x] 本地进程外子服务：ARDY Mini（20fps 文本动作 + 双臂 2-Bone IK）、Audio2Face-3D v2.3.1 Claire（10 维原生情绪 + 流式口型）、MediaPipe 精捕 + VMC/OSC
- [x] 画中世界（Mode 2）：MoGe-2 原生深度 + RGB 轮廓贴边遮挡细化 + 地面二次曲面 + 2D 可走网格 + A* 避障走去坐下

---

## Stage 5：Harness 大脑底座与「虚拟生命」闭环（已完成主线闭环）
- [x] **控制桥与统一日志**：`LocalControlServer.swift`（Apple `Network.framework`）+ `services/bridge/alicepet_mcp.py`（标准 MCP，可复用到任意外部 Harness / 网关）
- [x] **ZeroClaw 底座与两段式记忆**：接入 `ZeroClaw 0.9.0`（Rust <5MB）+ AnySoul 风格 Heartbeat / `TODO.md` + Animama 两段式 Markdown 记忆（`SOUL.md` / `PROFILE.md` / `MEMORIES.md` / `episodes/`）
- [x] **连续情绪惯性状态机**：`EmotionStateMachine.swift`（8 组对立情绪惯性与自然衰减）
- [x] **VLM 自主观察与 ARDY 联动验收**：LLM 主动看图观察房间、更新情绪与表情、触发 ARDY 20fps 动作并记录当日情景记忆

---

## Stage 6：表现力深化与语音闭环（下一阶段）
- [ ] **标准 `.vrma` 动画加载播放** + **说话手势节拍**（在 Layer 1 头部微动之上补上程序化手臂/手势节拍与情绪微体态）
- [ ] **画中模式空透明房间模式**（背景可设为空，便于纯透明底观察 ARDY 自由动作）
- [ ] **本地语音通话闭环**：`SenseVoice-Small` / `Parakeet` STT + Pipecat `Smart Turn` 语义断句 + `Pocket TTS` / 在线 TTS → Audio2Face-3D 实时口型
- [ ] **动态帧率省电策略**（互动 60fps / 静止待机 24–30fps）与更多通用 VRM 模型适配
- [ ] **远期灵感池**：Mode 3 原生对话窗、Mode 4 状态栏小工具、XR Animator 画中互动、参考 Folia 的后台音乐状态联动动画
