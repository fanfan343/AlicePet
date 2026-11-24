# VRM项目 · AlicePet

macOS（Apple Silicon）原生的 VRM 3D 桌面伴侣项目，基于 Swift + Metal（`VRMMetalKit`）。

## 目录结构

- `AlicePet/`：SwiftPM macOS 桌面应用源码（透明无边框置顶窗体、4× MSAA Metal 渲染、鼠标交互、摄像头捕捉与 ARKit 52 项 Perfect Sync 驱动）
- `AlicePet/build_app.sh`：构建并签名 `AlicePet.app`
- `AlicePhil(VRM) 1.01/` & `降采样模型/`：本地 VRM 模型资源（已通过 `.gitignore` 排除在版本库之外）

## 运行与打包

```bash
cd AlicePet
./build_app.sh
open ../AlicePet.app
```

日志位置：`~/Library/Logs/AlicePet/alicepet.log`（亦可通过菜单「打开日志文件夹」查看）。
