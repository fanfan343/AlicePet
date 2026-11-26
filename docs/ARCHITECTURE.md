# AlicePet · 虚拟生命（Virtual Being）与 Harness 架构设计

原则：**原生极轻、大脑与身体彻底解耦、协议统一、全链路可观测。**

---

## 1. 四种产品形态（4 Modes）

| 模式 | 形态与定位 | 用户交互边界 | 当前状态 |
|---|---|---|---|
| **Mode 1 · 普通桌宠模式** | 透明无边框置顶窗，陪伴挂机 | 用户可轨道旋转、平移、滚轮缩放、双击复位 | 已实现 |
| **Mode 2 · 画中模式（Picture-World）** | 2D 插画/照片（或空透明房间）+ 硬件深度遮挡 + A* 避障；VRM 与房间场景交互 | **用户仅滚轮缩放、中键平移观察、双击复位**；一切主动移动/就座/动作均由 LLM 决策，空闲时由程序维持 | 已实现（空房间背景与 VRMA 后续补齐） |
| **Mode 3 · 原生对话 UI** | Apple 原生独立 Chat 窗口（非 WebView） | 传统图文/语音对话界面 | 预留位置，暂不做 |
| **Mode 4 · 纯状态栏模式** | 无桌面窗口，仅 macOS 顶部状态栏图标 | 后续可挂载快捷小工具（Alice 的 Todo、剪贴板、AI 套餐额度、系统状态等） | 预留位置，后续扩展 |

---

## 2. 总体分层架构（大脑 / 记忆 / 协议 / 身体）

```text
                外部入口（手机 IM / 终端 / 其他 Harness）
                                  │
                                  ▼  (共享同一份 ZeroClaw Session & 记忆)
┌──────────────────────────────────────────────────────────────────────┐
│  大脑与网关底座：ZeroClaw（原生 Rust 单二进制，<5 MB 常驻）              │
│  ├── 多模型 Provider（当前测试：OpenCode Go · Claude Haiku）          │
│  ├── Heartbeat（AnySoul 风格低频自主心跳：看房间/推进 Todo/整理记忆）   │
│  ├── 记忆与人格文件系统（soul/，纯 Markdown，零向量库）                 │
│  │     • SOUL.md        灵魂、人设、行动准则                           │
│  │     • PROFILE.md     用户画像与偏好                                │
│  │     • MEMORIES.md    长期关系记忆（带热度与时间）                    │
│  │     • TODO.md        她自己的自主计划与待办                         │
│  │     • episodes/      每日情景流水账（两段式写入 → 定期折叠反思）      │
│  └── MCP Client / Gateway（挂载 alicepet_mcp）                        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  标准 MCP（stdio JSON-RPC）
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  极薄桥接层：services/bridge/alicepet_mcp.py（纯标准库，无外部依赖）     │
│  • 既服务 ZeroClaw，也可被任何其他支持 MCP 的 Harness 挂载              │
│  • 冲突仲裁原则：任何动作/状态冲突均以 AlicePet 本体当前状态为准         │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  127.0.0.1:39550 本地 JSON HTTP
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  身体与感官层：AlicePet（纯原生 Swift + Metal + VRMMetalKit）           │
│  ├── LocalControlServer（Apple Network.framework，零第三方依赖）       │
│  ├── EmotionStateMachine（8 组对立情绪惯性状态机，自然衰减回基线）       │
│  ├── AnimationMixer（Body SLERP + Head/手势节拍 + 表情/口型三轨解耦）   │
│  ├── SceneActionController + PictureWorldRenderer（硬件深度 + A*）     │
│  ├── 按需进程外重 AI 子服务（不开 0 显存）：ARDY / A2F-3D / MediaPipe   │
│  └── 统一日志：~/Library/Logs/AlicePet/alicepet.log                   │
│        ([App] / [Control] / [Emotion] / [MCP] / [Brain] / [Scene])    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心机制约定

### 3.1 两段式 Markdown 记忆（Animama × AnySoul）
1. **即时记录**：每轮交互中，重要事件先以单行时间戳笔记写入 `soul/episodes/YYYY-MM-DD.md`，零数据库开销。
2. **周期折叠与热度衰减**：由 ZeroClaw Heartbeat（或手动触发）定期审阅当日流水账，提炼更新到 `soul/PROFILE.md`、`soul/MEMORIES.md`（常用记忆升温，长期未提及的自然归档）与 `soul/TODO.md`。
3. **透明可回滚**：所有记忆文件均为人类可读的纯 Markdown，可随时人工编辑或通过 Git 回滚。

### 3.2 连续情绪惯性状态机（`EmotionStateMachine.swift`）
- 采用 **8 组两两对立的对话情绪轴**（`joy↔sadness`、`affection↔rejection`、`excitement↔calm`、`curiosity↔boredom`、`pride↔embarrassment`、`amusement↔annoyance`、`gratitude↔anger`、`hope↔anxiety`）。
- 收到情绪脉冲 `(emotion, intensity)` 时，抬升目标情绪并压低其对立情绪，同时随时间按半衰期平滑回落至温和基线，输出给 `ExpressionMixer` 与 `Audio2Face-3D` 10 维情绪向量。

### 3.3 本地控制接口（`127.0.0.1:39550`）
- `GET  /health` — 服务存活与端口检查
- `GET  /state` — 完整 `AvatarSceneState` + 情绪状态 + 当前模式 + 房间锚点列表
- `POST /observe` — 截取当前实时画面 PNG，返回实时图路径、房间原图 `bg.png` 路径与状态摘要（供 VLM 视觉观察）
- `POST /look` — 设置姿态 / 手势 / 表情 / 换装 / 注入情绪脉冲
- `POST /ardy` — 启动、下发实时动作提示词或停止 ARDY 20fps 全身动作
- `POST /scene/walk-sit` / `POST /scene/stand-up` / `POST /scene/load` / `POST /scene/exit` — 房间切换与自主走动就座
- `POST /scene/anchors` — 接收 VLM 视觉定位的 2D 归一化坐标 `(u, v)` + 标签 + 朝向，由 App 通过 `depth.bin` 反投影为 3D 锚点并保存进 `scene.json`
