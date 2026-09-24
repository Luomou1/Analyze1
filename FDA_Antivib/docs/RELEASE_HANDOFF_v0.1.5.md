# v0.1.5 发布与维护交接

## 交付范围

目标仓库为 [Luomou1/Analyze1](https://github.com/Luomou1/Analyze1)，版本标签为 `v0.1.5`，源码在 `main`。发布页面为 [v0.1.5](https://github.com/Luomou1/Analyze1/releases/tag/v0.1.5)，Windows 安装包名称为 `AnalysisProgram-0.1.5-setup.exe`。版本号由 `app/__init__.py` 与 `pyproject.toml` 同步维护。

此版本整合此前工作区中的 GFDA 独立重建、标定保存/加载、扫描诊断和导出，以及全分辨率三维后台网格准备、按需绘图与二维坡度光照。此次发布工作补齐仓库首页、更新应用说明并保存研究交接，没有将尚未通过实测的研究原型接入产品。

## 已知限制与研究衔接

GFDA 仍属实验功能，两组真实扫描数据未达到抗振改善验收要求。完整数据与失败方案、指标口径、下一轮验收条件见 [GFDA 研究交接](GFDA_交接_2026-09-24.md)。不能把软件回归通过解读为 GFDA 实测问题已解决，也不能声称与 MetroPro 全流程等价。

GitHub 源码包含产品模块、测试及现有离线测试工件，但不包含根 `tmp/` 下的研究脚本/反汇编/结果、大型原始扫描和原厂程序。研究交接中的 `D:/`、`E:/` 路径属于原开发机。继续真实数据研究需另行取得这些证据和原始数据；不能仅凭克隆仓库声称重现历史实测。

三维显示使用全分辨率网格；后台仅准备网格，GPU 上传与首次绘制仍在界面线程，可能短暂停顿。大型数据的耗时和内存取决于图像规模、硬件与模式。未完成所有显卡环境、真实交互安装升级和仪器精度验收。

## 验证记录

2026-09-24 使用 Windows / Python 3.11.6 / PyInstaller 6.20.0 执行验证：完整 `python -m pytest -q --tb=short` 为 **673 passed in 145.92s**，退出码 0；`python -m compileall -q app main.py`、文档本地链接与版本一致性检查、`git diff --check` 均通过。

第一次仅设置 `QT_QPA_PLATFORM=offscreen` 的全套测试在 VTK interactor 初始化处发生访问异常。补设 `PYVISTA_OFF_SCREEN=true` 后，显示定向回归 18 项通过，上述完整 673 项回归通过；没有为绕过失败修改产品代码。

PyInstaller 构建成功。构建脚本内的 EXE 导入检查在受限沙箱中超过 60 秒，脚本因此没有生成安装器；随后在沙箱外单独执行同一 EXE 的 `--smoke-test-imports`，设置 `PYINSTALLER_RESET_ENVIRONMENT=1`，退出码为 0。此后执行 `ISCC.exe /DMyAppVersion=0.1.5 packaging/data_analysis.iss`，安装器构建成功，退出码 0。未执行实际安装/升级向导验收。

安装包大小为 **212327411 字节**，SHA-256 为 `513c146ba8ef0fc5ffd320c571dc37cf6cb6fad3ad2b7aba037c24d6f9284da7`。远端发布后需核对 GitHub 附件 digest 与此值一致。原始测试及构建日志保存在原开发机 `tmp/release-v0.1.5-*.log`，未作为源码提交。

## 复现与维护

从仓库根目录按 README 创建 Python 3.11 虚拟环境并安装 `./FDA_Antivib[test]`，进入 `FDA_Antivib` 后执行：

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYVISTA_OFF_SCREEN = 'true'
python -m pytest -q --tb=short
python -m compileall -q app main.py
pwsh -File scripts/build_windows_exe.ps1
```

以上 `python` 必须指向已安装依赖的环境。构建额外需要 PyInstaller、PowerShell 7 和 Inno Setup 6。安装包生成后复制为英文附件名，上传 GitHub Release，并使用 `Get-FileHash -Algorithm SHA256` 与 `gh api repos/Luomou1/Analyze1/releases/tags/v0.1.5` 返回的附件 digest、size 核对。本次保留本地构建产物，不调用包含清理动作的上传脚本。

此次从 v0.1.4 所在提交 `2c33ca8` 追加提交，不重写历史。旧安装包可从 v0.1.4 Release 下载；如遇新版本问题可重新安装旧版进行对照，保留原始数据、导出结果及校正文件。后续修复应使用新版本号和新标签，避免覆盖已发布附件。

## 审查范围

按 requesting-code-review 清单本地复核版本一致性、重建入口到 GFDA 的参数传递、全帧路径、标定与诊断导出、后台绘图结果失效机制、打包依赖、文档链接及实测限制。按用户规则未派发子代理。本次只调整版本元数据和文档，不开展算法重构；软件回归和打包冒烟不替代人工安装流程及真实样品验收。
