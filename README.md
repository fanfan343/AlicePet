# VRM项目 · AlicePet

macOS（Apple Silicon）原生的超轻量 VRM 3D 桌面伴侣项目，基于 Swift + Metal（`VRMMetalKit` / `GLTFMetalKit`）。

## 目录结构

- `AGENTS.md`：AI Vibe Coding 协作规则、架构边界与强制验收流程
- `AlicePet/`：SwiftPM macOS 原生主应用源码与打包脚本
- `tools/`：资产提取工具（`extract_alice_assets.py`）与一键视觉回归脚本（`verify_visuals.sh`）
- `docs/`：分阶段路线图（[ROADMAP.md](docs/ROADMAP.md)）与设计文档
- `profiles/`：预留——通用 VRM 模型适配配置目录
- `services/`：预留——进程外 AI 子服务目录（`motion/`、`llm/`、`tts/`）
- `snapshots/`：自动生成的视觉回归截图目录（`latest/` 与 `baseline/`，图片不入 Git）
- `AlicePhil(VRM) 1.01/` & `降采样模型/`：本地模型与道具资产（已通过 `.gitignore` 排除）

## 常用命令

```bash
# 1. 构建并签名 Release 版 AlicePet.app（推荐日常挂机，低 CPU / 省电）
./AlicePet/build_app.sh

# 如需构建 Debug 版快速调试
./AlicePet/build_app.sh debug

# 2. 启动桌面伴侣
open ./AlicePet.app

# 3. 跑一键视觉回归截图（自动生成多场景图 + 总览拼图 snapshots/latest/contact_sheet.png）
./tools/verify_visuals.sh
```

日志位置：`~/Library/Logs/AlicePet/alicepet.log`（亦可通过菜单栏「打开日志文件夹」查看，包含定期性能基线记录）。
