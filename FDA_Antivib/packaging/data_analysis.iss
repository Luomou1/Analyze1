#define MyAppName "分析程序"
#ifndef MyAppVersion
  #error MyAppVersion must be supplied by build_windows_exe.ps1
#endif
#define MyAppPublisher "Luomou1"
#define MyAppExeName "分析程序.exe"

[Setup]
AppId={{A1D93E87-9835-4D7F-B293-4D77BB967844}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
; 升级时也显示目录页，并将原目录作为可修改的默认值。
DisableDirPage=no
UsePreviousAppDir=yes
DefaultGroupName={#MyAppName}
UsePreviousGroup=no
DisableProgramGroupPage=yes
OutputDir=..\dist\installer
OutputBaseFilename=分析程序-{#MyAppVersion}-setup
SetupIconFile=..\assets\app_icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加快捷方式："; Flags: unchecked

[Files]
Source: "..\dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[InstallDelete]
; 同目录升级时清理旧名称程序与由旧安装器创建的快捷方式。
Type: files; Name: "{app}\数据分析.exe"
Type: files; Name: "{autoprograms}\数据分析\数据分析.lnk"
Type: dirifempty; Name: "{autoprograms}\数据分析"
Type: files; Name: "{autodesktop}\数据分析.lnk"

[Run]
; 在原用户进程内设置标志，避免安装器提权前后的环境隔离使设置失效。
Filename: "{cmd}"; Parameters: "/D /S /C ""set ""PYINSTALLER_RESET_ENVIRONMENT=1"" && ""{app}\{#MyAppExeName}"""""; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent runhidden
