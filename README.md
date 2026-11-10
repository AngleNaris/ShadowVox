# ShadowVox

ShadowVox 是一个本地运行的中文歌声修音工作台。导入演唱干声和带原唱的原曲后，可校正音高和节奏，再与分离出的伴奏混音。

- 干声分离、降噪、削波修复、音高与节奏校正
- 前后试听、实时草稿、批量处理和版本化导出
- Windows 桌面界面、命令行和离线处理报告

## 从源码运行

使用 Windows 和 Python 3.13。安装依赖后，准备模型及 FFmpeg、Rubber Band、降噪组件和 ZL 插件。模型和工具的来源见[第三方组件说明](THIRD_PARTY.txt)。

```powershell
python -m pip install -r requirements-desktop.txt
python desktop.py
```

命令行支持 `python run.py desktop --help` 和 `python run.py batch --help`。更多用法见[桌面与 CLI 文档](docs/desktop-v1.md#gui-与-cli-覆盖核对)。

作者 **繁星之子卡萨蒂亚**。
