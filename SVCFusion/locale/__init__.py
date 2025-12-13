import os
import importlib.util


def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


locale_dict = {}
text_to_locale = {}

for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename not in ["__init__.py", "base.py"]:
        file_path = os.path.join(os.path.dirname(__file__), filename)
        module_name = os.path.splitext(filename)[0]
        module = load_module_from_file(module_name, file_path)

        if (
            hasattr(module, "_Locale")
            and hasattr(module, "locale_name")
            and hasattr(module, "locale_display_name")
        ):
            _Locale = getattr(module, "_Locale")
            locale_name = getattr(module, "locale_name")
            locale_display_name = getattr(module, "locale_display_name")

            locale_dict[locale_name] = _Locale
            text_to_locale[locale_display_name] = locale_name

# fallback: 始终保证中文可用
if "zh-cn" not in locale_dict:
    # 兼容性处理，如果没有zh-cn，尝试zh_CN
    if "zh_CN" in locale_dict:
        locale_dict["zh-cn"] = locale_dict["zh_CN"]
    else:
        # fallback失败，什么都不做
        pass


def get_locale(lang: str):
    # 优先返回指定语言，否则返回中文
    if lang in locale_dict:
        return locale_dict[lang]
    return locale_dict.get("zh-cn") or next(iter(locale_dict.values()))


__all__ = ["locale_dict", "text_to_locale", "get_locale"]
