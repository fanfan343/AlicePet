# AlicePet · AI 协作与开发规矩（AGENTS.md）

本项目是面向 macOS（Apple Silicon）的开源原生超轻量 3D VRM 桌面伴侣（Swift + Metal / `VRMMetalKit`），仓库地址：[fanfan343/AlicePet](https://github.com/fanfan343/AlicePet)。
项目以 Vibe Coding 方式推进开发，**所有修改代码的 AI 必须严格遵守以下规则**。

---

## 1. 核心目标与技术原则

1. **原生 Mac、极致省资源**：主 App 保持纯原生 Swift + Metal（与 Apple `Network.framework`），严禁引入 Electron / WebView / Unity 等重型运行时；渲染与面捕优先考虑低 CPU / GPU 占用与动态帧率策略。
2. **主程序轻量、重 AI 解耦**：主 App 只负责渲染、三层动画混合、画中世界、情绪状态机、本地控制桥与标准驱动接口；重 AI 子服务（**ARDY Mini 20fps 动作生成**、**MoGe-2 单图深度估计**、**Audio2Face-3D / LAM 表情口型**、**MediaPipe 动捕**、**ZeroClaw 大脑与 MCP 桥**）一律放入 `services/` 按需启停，关闭即 0 显存占用。
3. **代码模块化，禁止堆「上帝文件」**：单个 `.swift` 文件尽量控制在 350 行以内；职责按渲染、动画混合、情绪状态机、画中世界、控制桥、菜单 UI 清晰拆分。
4. **通用 VRM 优先，专属配置外置**：主程序只依赖 VRM 0.0 / 1.0 与 `.vrma` 标准规范；特定模型（如 Alice）的节点名、换装、发型与道具规则外置到配方/配置中，更换其他 VRM 模型时必须平滑降级、不崩溃。
5. **先查现成轮子，不重复造轮子**：遇到通用问题（如碰撞物理、IK、`.vrma` 动画加载、音视频/推理管线）时，第一时间先核查现有依赖（如 `VRMMetalKit` / Apple 原生框架）与成熟开源轮子，能直接复用就不手写替代品。
6. **治本不治标**：排查 Bug 或体验问题时必须追到底层根因（模型原始输出、坐标系、渲染管线、物理边界等），严禁用表面参数硬凑或掩盖症状。
7. **范式优先，禁止启发式钻牛角尖**：不靠在小样本上反复叠加阈值/特判硬凑通用性；若某条路线连续打补丁仍不稳定，或正在把附属功能做成重型子系统，必须立刻停手与项目负责人重新讨论范式与优先级，优先选择更简单、普适、可删除的方案。
8. **开源规范与参考致谢持续维护**：本项目已公开开源。任何时候参考、集成或受启发于外部开源项目、论文或模型（如 `VRMMetalKit`、`ARDY`、`MoGe-2`、`Audio2Face-3D`、`ZeroClaw`、`Animama`、`AnySoul` 等），**必须同步登记到根目录 [README.md](README.md) 的「参考项目与致谢」章节**；严禁提交任何 API Key、个人隐私日志（`soul/episodes/*.md`）、第三方受版权保护的模型文件（`.vrm` / `.glb`）或大模型权重。

---

## 2. 目录结构与边界

| 目录 | 用途 | Git 跟踪 | 注意事项 |
| --- | --- | --- | --- |
| `AlicePet/` | macOS 原生 SwiftPM 主应用源码与打包脚本 | 是 | 严禁把大型模型或临时截图放进这里 |
| `services/` | 进程外按需 AI 子服务（`ardy/`、`scene/`、`tts/`、`motion/`、`bridge/`） | 仅源码 | 权重文件、`.venv`、缓存一律不入 Git |
| `soul/` | 两段式 Markdown 记忆模板（`SOUL.md` / `PROFILE.md` / `MEMORIES.md` / `TODO.md`） | 仅模板 | `soul/episodes/*.md` 日常流水账已由 `.gitignore` 忽略 |
| `tools/` | 资产提取脚本与自动化视觉回归脚本 | 是 | 尽量零外部依赖（原生 Swift / Shell / 标准库 Python） |
| `docs/` | 架构文档（`ARCHITECTURE.md`）、路线图（`ROADMAP.md`）与精选展示图（`docs/assets/`） | 是 | 仅允许在 `docs/assets/` 放置轻量精选展示图（<500KB） |
| `profiles/` | 各 VRM 模型的适配配置（不含版权二进制） | 是 | 仅存放 JSON/YAML 配置与说明 |
| `snapshots/` | 视觉回归截图（`latest/` 与 `baseline/`） | 仅占位 | PNG 产物由 `.gitignore` 忽略，不提交二进制图 |
| `AlicePhil(VRM) 1.01/` | 本地第三方授权原始模型资产 | **否** | **只读，严禁移动、删除或修改** |
| `降采样模型/` | 本地优化后的 VRM 与独立道具/配方 | **否** | 保持原路径稳定，勿随意重命名 |

---

## 3. 强制验收闭环（改完必须编译 + 跑验证）

每次修改 `AlicePet/` 代码后，**在向用户汇报「已完成」之前，必须完整走完以下闭环**：

1. **必须编译通过**：
   - 开发迭代快速检查：`cd AlicePet && swift build`
   - 交付可运行 App：`./AlicePet/build_app.sh`（默认构建 `release` 版；如需调试版运行 `./AlicePet/build_app.sh debug`）
   - 任何编译错误或新增严重警告必须先自行修完。
2. **必须跑视觉回归验证**：
   - 执行 `./tools/verify_visuals.sh`（若改动影响渲染、姿态、表情、换装、道具或模型加载）。
   - 检查输出截图（`snapshots/latest/contact_sheet.png` 及关键单图），确认无黑屏、无明显穿模、无材质丢失、目标改动真实生效、既有功能无回归。
3. **必须检查日志**：
   - 运行验证后检查命令输出及 `~/Library/Logs/AlicePet/alicepet.log`，确认无新增非预期 `ERROR`。
4. **向用户汇报必须给结论**：
   - 简洁说明：改了什么、编译结果、截图验证结果（必要时展示总览图）、潜在风险或待用户拍板事项。

---

## 4. Git 提交与 GitHub 看板同步规范

- **一件事一个小提交**：每次只聚焦一个明确目标（不要把多个不相关功能揉成一笔巨型提交），方便随时按功能回退。
- **使用清晰前缀**：`feat:` / `fix:` / `refactor:` / `perf:` / `docs:` / `chore:`。
- **提交前确认 `git status` 与无密钥泄漏**：确保没有把 `.vrm`、临时 `.png`、`.build/`、模型权重、日志或 API Key 误提交进仓库。
- **远端同步与进度看板维护**：
  - 仓库已关联 `origin/main`（`https://github.com/fanfan343/AlicePet`），Git 不会在保存文件时自动推送，需在验收通过并提交后执行 `git push origin main`。
  - 每完成一个阶段性功能或里程碑时，同步更新：
    1. [docs/ROADMAP.md](docs/ROADMAP.md) 与 [README.md](README.md) 进度表（及新增参考项目的致谢条目）；
    2. 对应的 GitHub Issue 状态、Milestone 进度与 [Projects v2 看板](https://github.com/users/fanfan343/projects/1)（`Status` / `Stage`）。
