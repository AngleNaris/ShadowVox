"""仅载入 VST3、显示原生界面并保存状态；不向插件发送音频。"""
import json
from pathlib import Path


def _load_state(plugin, state_path):
    if state_path.is_file():
        plugin.raw_state = state_path.read_bytes()
    parameters_path = state_path.with_suffix(".params.json")
    if parameters_path.is_file():
        saved = json.loads(parameters_path.read_text(encoding="utf-8"))
        parameters = plugin.parameters
        for name, value in saved.items():
            if name in parameters:
                parameter = parameters[name]
                parameter.raw_value = value
                if abs(parameter.raw_value-value) > .002:
                    raise RuntimeError(f"插件参数无法恢复：{name}")


def _save_state(plugin, state_path):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = {name:float(parameter.raw_value) for name,parameter in plugin.parameters.items()
                  if parameter.is_automatable}
    temp = state_path.with_suffix(".tmp")
    temp.write_bytes(plugin.raw_state)
    temp.replace(state_path)
    parameters_path = state_path.with_suffix(".params.json")
    temp = state_path.with_suffix(".params.tmp")
    temp.write_text(json.dumps(parameters,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(parameters_path)


def open_editor(plugin_path, state_path, close_event=None):
    from pedalboard import load_plugin

    plugin_path = Path(plugin_path)
    if plugin_path.suffix.lower() != ".vst3" or not plugin_path.exists():
        raise ValueError("请选择存在的 VST3 插件文件或文件夹。")
    state_path = Path(state_path)
    plugin = load_plugin(str(plugin_path))
    _load_state(plugin,state_path)
    plugin.show_editor(close_event)
    _save_state(plugin,state_path)
