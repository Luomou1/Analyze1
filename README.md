# Analyze1 · 分析程序

用于白光扫描干涉数据重建、平面分析和台阶分析的 Windows 桌面程序。界面基于 PySide6，数值计算使用 NumPy/SciPy/FINUFFT，二维和三维显示使用 Matplotlib 与 PyVista。运行时不需要安装 MetroPro，也不调用原厂 EXE/DLL 计算。

当前版本 **v0.1.5**。GFDA 已提供独立抗振计算、扫描坐标诊断和校正文件管理，但真实数据改善尚未达到验收要求；该功能应作为实验功能使用。项目不宣称完整复现原厂算法或取得计量精度认证。

## 下载与安装

在 [GitHub Releases](https://github.com/Luomou1/Analyze1/releases/latest) 下载 `AnalysisProgram-0.1.5-setup.exe`，运行安装向导选择目录。支持 Windows 10 1809 及以上和 Windows 11 x64；安装包自带 Python 与运行依赖。程序名称和快捷方式为“分析程序”，桌面快捷方式可选。程序内的更新检查指向本仓库最新正式 Release。

## 从源码运行

已在 Windows / Python 3.11 环境验证；项目声明 Python >=3.11。以下命令在 PowerShell 中执行，虚拟环境无需激活：

```powershell
git clone https://github.com/Luomou1/Analyze1.git
cd Analyze1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e './FDA_Antivib[test]'
cd FDA_Antivib
..\.venv\Scripts\python.exe main.py
```

依赖以 [pyproject.toml](FDA_Antivib/pyproject.toml) 为准。三维窗口需要可用的 OpenGL 图形环境；无头测试结果不代表所有显卡上的显示兼容性。大型扫描会同时占用图像立方体、计算中间数组和全分辨率三维网格的内存。

## 基本操作与单位

选择同一次扫描的图像目录，支持 JPG、TIF/TIFF、PNG、BMP。按采集格式选择强度模式，正确填写扫描步长；单位为 **μm**，例如 50 nm 输入 `0.05`。不要把相机位深、12 位左对齐和普通 16 位数据混用。

选择重建工作流后开始分析，查看高度图、三维图与像素频谱，按需导出结果。公共 K0 预览不是正式计算的必需步骤。普通模式自动确认有效扫描范围；GFDA 保留全部采集帧。输出高度单位为 **nm**，无效点保留为 NaN。

平面和台阶分析是独立入口，支持去形状、尖峰排除、补洞、空间/频率滤波、区域测量和报告导出。频率滤波及部分高级拟合需要样品面横向标定，单位为 **μm/pixel**，与轴向扫描步长不同。台阶可选三点调平；框选均值差不依赖显示分层标签。

## 重建模式

| 模式 | 输出与适用说明 |
| --- | --- |
| Normal | 相位斜率高度，提供基础 FDA 重建。 |
| High | 独立相位偏差直方图、局部载频细化与区域残差处理。 |
| High1G | 公共 K0 相位高度和整数级次选择，不执行区域连接；名称为本项目定义。 |
| High 2G | 整数相位域区域连接，当前为单材料、无色散标定路径。 |
| GFDA（实验） | 初始 FDA/High 2G、扫描运动估计、非均匀频谱与 High 2G 输出。 |

GFDA 需要足够空间条纹，当前只支持共同轴向正向扫描及相邻相位增量小于半周期的适用域。错误的名义步长不会自动校正；低调制度、饱和、秩亏或不可观测扫描可能报错。两组实测中 High 2G 的调平 Sq 约为 3.135/3.042 nm，当前 GFDA 约为 3.037/3.055 nm，尚不能认定有效改善。研究原型的结果、否决方案及后续验收条件见交接文档。

## 文档与目录

[详细使用说明与算法边界](FDA_Antivib/README.md) 包含 K0、解包裹、GFDA 校正、表面分析参数及原厂对照范围。[GFDA 研究交接](FDA_Antivib/docs/GFDA_交接_2026-09-24.md) 记录实测、失败方案、证据路径和待办；[v0.1.5 发布与维护交接](FDA_Antivib/docs/RELEASE_HANDOFF_v0.1.5.md) 记录发布范围、验证与维护方法。

```text
FDA_Antivib/
  main.py              桌面程序入口
  app/gui/             窗口、控件与后台任务
  app/pipeline/        输入、扫描范围、会话与导出
  app/reconstruction/  FDA、High、High1G、High 2G、GFDA
  app/plotting/        二维/三维绘图
  tests/               数值、界面、输入输出与发布回归
  docs/                设计记录和交接文档
  packaging/           PyInstaller 与 Inno Setup 配置
  scripts/             构建、发布和辅助脚本
```

根目录 `tmp/`、`private/`、输出目录和构建产物由 Git 忽略。交接文档引用的本机原始采集数据、研究脚本及反汇编不在此仓库；克隆后可运行产品测试，但不能仅凭仓库复现全部历史实测实验。原厂离线测试工件的来源和范围记录在 `tests/data/` 的对应 JSON 中。

## 测试

在 `FDA_Antivib` 目录执行：

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYVISTA_OFF_SCREEN = 'true'
..\.venv\Scripts\python.exe -m pytest -q --tb=short
..\.venv\Scripts\python.exe -m compileall -q app main.py
```

完整回归覆盖重建、GFDA、表面分析、GUI、导出和发布脚本。无头运行需同时设置 Qt 和 PyVista 的上述环境变量，避免 VTK 尝试初始化交互窗口；正常桌面使用时不要设置这些变量。原厂局部函数及部分应用整链的离线对照，只证明已覆盖的输入与分支；测试通过不等于仪器精度认证。

## 构建 Windows 安装包

构建需要 PowerShell 7、PyInstaller 和 Inno Setup 6，`python` 应指向已安装项目依赖的环境。从 `FDA_Antivib` 执行：

```powershell
python -m pip install pyinstaller
pwsh -File scripts/build_windows_exe.ps1
```

脚本生成 `dist/分析程序.exe`，执行 `--smoke-test-imports` 检查，再生成 `dist/installer/分析程序-0.1.5-setup.exe`。上传时使用英文文件名 `AnalysisProgram-0.1.5-setup.exe`。新版本必须同步更新 `app/__init__.py` 和 `pyproject.toml`，完成测试、源码提交与标签后上传安装包，并核对远端大小和 SHA-256。

## 问题反馈与许可状态

请在 [Issues](https://github.com/Luomou1/Analyze1/issues) 提供程序版本、Windows 版本、模式、步长/位深、复现操作及日志；数值问题还需注明掩膜、调平和滤波设置。提交可共享的最小样例时请保留采集参数。

仓库当前未提供 LICENSE 文件，不应推定为采用某种开源许可证；第三方依赖的许可分别以各自项目为准。
