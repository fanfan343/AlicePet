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

## Stage 3：动作系统深化（VRMA + Kimodo / ARDY）
- [x] 定义统一的 `MotionSource` 协议，并实现原版 Unity Humanoid 肌肉动画提取器（`tools/extract_alice_animations.py`）与实时重定向播放器（`UnityMuscleRetargeter` + `AuthorAnimationPlayer`），还原作者原版 Stand / Sit / Crouch / Prone / 16.67s AFK 循环动作
- [ ] 支持加载播放标准 `.vrma` 动画文件与姿态间平滑过渡（Crossfade）
- [ ] 在 `services/motion/` 中以独立本地子服务方式接入 **Kimodo / ARDY** 实时动作生成与骨架重定向

## Stage 4：面捕与上半身捕捉进阶
- [x] 抗歪头旋转不变的眼睛开合比（EAR）计算
- [x] 中性表情与坐姿一键校准、双目平均瞳孔视线追踪、皱眉/脸颊联动
- [x] 原生 Apple Vision 3D 上半身骨架追踪（`VNDetectHumanBodyPose3DRequest` + `BodyTrackingSolver` 1€ 滤波，驱动脊柱/胸口/双臂联动）
- [ ] 可选支持 MediaPipe 52 通道精捕子服务（`services/`）或 iPhone VMC/OSC 52 通道实时输入

## Stage 5：AI 大脑与语音口型（LLM + TTS）
- [ ] 在 `services/llm/` 中接入可切换的在线 / 本地 LLM 对话服务
- [ ] 在 `services/tts/` 中接入在线 / 本地 TTS 语音合成与实时音频 LipSync 口型驱动
- [ ] 情绪标签自动联动表情配方、手势与待机姿态
