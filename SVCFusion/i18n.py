from SVCFusion.config import system_config
from .locale.base import Locale
from .locale import text_to_locale, get_locale

lang = text_to_locale.get(system_config.pkg.lang, "zh-cn")

"""
国际化
"""
I: Locale = get_locale(lang)  # noqa: E741
