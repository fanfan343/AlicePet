# AlicePet 路线图（ROADMAP）

原则：**一步一步推进，先打牢渲染与通用模型适配，再逐步叠动作流、面捕与 AI 交互。**

## Stage 1：工程护栏与基础闭环（当前进行中）
- [x] macOS 透明无边框置顶窗 + `VRMMetalKit` 4× MSAA 渲染
- [x] 基础交互（拖窗、轨道旋转、平移、缩放、菜单栏控制）
- [x] Mac / iPhone 连续互通相机面捕 + 1€ 自适应滤波
- [x] GLTF 独立道具叠加、AFK 姿态、手势、换装与日志系统
- [x] 建立项目 `AGENTS.md`、规范预留目录、一键视觉回归脚本与 `release` 构建
- [x] 轻量级性能采样日志（FPS / 帧耗时 / 内存基线）

## Stage 2：架构拆薄与渲染打磨（进行中）
- [x] 将 `PetController.swift` 拆分为独立模块：骨骼姿态（`PoseController` + `MotionSource` 协议）、表情分层（`ExpressionMixer`）、模型适配（`ModelProfile`）
- [x] 把 Alice 专属节点/换装/发型/道具偏移外置到 `profiles/alice.json`，主程序支持通用 VRM 降级
- [x] 修复切姿态 SpringBone 炸毛/发箍飞起（`resetPhysics` + `warmupPhysics`）、表情菜单勾选态、换装菜单动态检测、记住上次加载的模型路径
- [x] 打磨 MToon 赛璐璐光影（4 套预设 + 描边可选 + 恢复作者原色/自发光）、道具多灯均衡补光、消除 AFK 裙摆穿模
- [ ] 动态帧率省电策略（互动 60fps / 静止待机 24–30fps / 被遮挡降频或休眠）
- [ ] 接入 1–2 个非 Alice 的通用 VRM 模型验证跨模型兼容性

## Stage 3：动作系统深化（分层混合器 + ARDY Mini）
- [x] 定义统一的 `MotionSource` 协议，并实现原版 Unity Humanoid 肌肉动画提取器（`tools/extract_alice_animations.py`）与实时重定向播放器（`UnityMuscleRetargeter` + `AuthorAnimationPlayer`），还原作者原版 Stand / Sit / Crouch / Prone / 16.67s AFK 循环动作
- [x] 实现三层动画混合器 `AnimationMixer`（独立 VMC 通道隔离 + Layer 0 全身 0.35s SLERP Crossfade 过渡 + Layer 1 语音头部微动四元数叠加 + Layer 2 下脸口型 / 上脸情绪 / 视线与眨眼三轨解耦）
- [x] 在「测试功能 (Lab)」中以独立本地子服务方式接入 **ARDY Mini**（`services/ardy/`，文本驱动 20fps 自回归全身动作）与通用双臂比例 2-Bone IK / 防穿模保护
- [x] 接入 **NVIDIA Audio2Face-3D v2.3.1 Claire**（`services/tts/audio2face3d_vmc.py`，官方 ONNX + 52-D 缓存求解器，支持原生 10 维情绪与真实眉眼动态，并可在 Lab 面板与 LAM 切换对比）
- [x] 实现轻量「画中世界（形态 B）」原生 Metal 渲染器（`PictureWorldRenderer` + `services/scene/` MoGe-2 单图深度与地平面重建，支持透明桌宠 ↔ 2D→3D 房间硬件深度遮挡一键切换）
- [ ] 支持加载播放标准 `.vrma` 动画文件

## Stage 4：面捕与上半身捕捉进阶
- [x] VMC / OSC UDP 统一动捕接收器（默认端口 `39539` / `39540`，支持 52 通道 ARKit PerfectSync + 全身/十指骨骼，兼容外部发包端与本机服务）
- [x] 本地无界面 MediaPipe 精捕子服务（`services/motion/`：直接复用 `face_landmarker` + `pose_landmarker` + `hand_landmarker` 3 个 `.task` 模型，输出 52 通道表情 + 上半身 + 十指，菜单栏一键启停、关闭即终止进程 100% 释放摄像头）
- [x] 废弃旧版 Apple Vision 启发式摄像头捕捉与抖动体捕，统一收口到 VMC 管线与简洁菜单栏

## Stage 5：AI 大脑与语音口型（LLM + TTS）
- [ ] 在 `services/llm/` 中接入可切换的在线 / 本地 LLM 对话服务
- [ ] 在 `services/tts/` 中接入在线 / 本地 TTS 语音合成与实时音频 LipSync 口型驱动
- [ ] 情绪标签自动联动表情配方、手势与待机姿态
