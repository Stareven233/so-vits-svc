from SVCFusion.config import applyChanges
from SVCFusion.i18n import I
from SVCFusion.config import system_config
from SVCFusion.locale import text_to_locale
from SVCFusion.ui.Form import Form

from utils import Dummy; gr = Dummy()

from SVCFusion.ui.FormTypes import FormDict


class Settings:
    def get_save_config_fn(self, prefix):
        def fn(config, progress=None):
            tmp = {
                prefix: config,
            }
            applyChanges(
                "configs/svcfusion.json",
                tmp,
                no_skip=True,
            )
            gr.Info(I.settings.saved_tip)

        return fn

    def __init__(self):
        self.form: FormDict = {
            I.settings.pkg_settings_label: {
                "form": {
                    "lang": {
                        "label": I.settings.pkg.lang_label,
                        "type": "dropdown",
                        "info": I.settings.pkg.lang_info,
                        "choices": text_to_locale.keys(),
                        "default": lambda: system_config.pkg.lang,
                    },
                    "workdir_location": {
                        "label": I.settings.pkg.workdir_location_label,
                        "type": "textbox",
                        "info": I.settings.pkg.workdir_location_info,
                        "placeholder": "",
                        "default": lambda: system_config.pkg.workdir_location,
                    },
                },
                "callback": self.get_save_config_fn("pkg"),
            },
            I.settings.sovits_settings_label: {
                "form": {
                    "resolve_port_clash": {
                        "type": "checkbox",
                        "label": I.settings.sovits.resolve_port_clash_label,
                        "info": I.settings.sovits.resolve_port_clash_label,
                        "default": lambda: system_config.sovits.resolve_port_clash,
                    }
                },
                "callback": self.get_save_config_fn("sovits"),
            },
            I.settings.infer_settings_label: {
                "form": {
                    "msst_device": {
                        "type": "device_chooser",
                        "info": I.settings.infer.msst_device_label,
                    }
                },
                "callback": self.get_save_config_fn("infer"),
            },
        }

        self.triger = gr.Dropdown(
            label=I.settings.page,
            choices=self.form.keys(),
            value=I.settings.pkg_settings_label,
            interactive=True,
        )

        Form(
            triger_comp=self.triger,
            models=self.form,
            submit_btn_text=I.settings.save_btn_value,
        )
