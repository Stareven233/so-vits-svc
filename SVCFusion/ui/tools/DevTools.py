from os import system
import os
from utils import Dummy; gr = Dummy()

from SVCFusion.dlc import MetaV1, pack_directory_to_dlc_file


class DevTools:
    def pack_pretrain_dlc(self, directory_paths: str, model_name):
        import concurrent.futures

        def pack_directory(directory_path):
            meta: MetaV1 = {
                "version": "v1",
                "type": "pretrain",
                "attrs": {
                    "model": model_name,
                },
            }
            print(f"packing {directory_path}")
            os.makedirs("dev/dlc_packs", exist_ok=True)
            pack_directory_to_dlc_file(
                directory_path,
                meta,
                f"dev/dlc_packs/pretrain/{model_name}_{os.path.basename(directory_path)}.sf_dlc",
            )
            print(f"packed {directory_path}")

        directory_paths_list = directory_paths.split("\n")
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for directory_path in directory_paths_list:
                futures.append(executor.submit(pack_directory, directory_path))
            progress = gr.Progress()
            for future in progress.tqdm(concurrent.futures.as_completed(futures)):
                future.result()
        system(f'explorer "{os.path.abspath("dev/dlc_packs/pretrain/")}"')

    def pack_vocoder_dlc(self, directory_paths: str):
        import concurrent.futures

        def pack_directory(directory_path):
            meta: MetaV1 = {
                "version": "v1",
                "type": "vocoder",
                "attrs": {
                    "name": os.path.basename(directory_path),
                },
            }
            print(f"packing {directory_path}")
            os.makedirs("dev/dlc_packs", exist_ok=True)
            pack_directory_to_dlc_file(
                directory_path,
                meta,
                f"dev/dlc_packs/vocoder/{os.path.basename(directory_path)}.sf_dlc",
            )
            print(f"packed {directory_path}")

        directory_paths_list = directory_paths.split("\n")
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for directory_path in directory_paths_list:
                futures.append(executor.submit(pack_directory, directory_path))
            progress = gr.Progress()
            for future in progress.tqdm(concurrent.futures.as_completed(futures)):
                future.result()
        system(f'explorer "{os.path.abspath("dev/dlc_packs/vocoder/")}"')

    def fn_pack_pretrain_dlc(self):
        gr.Markdown("## 打包为预训练 DLC")
        pack_to_pretrain_dlc_input_path = gr.Textbox(label="输入目录路径(可批量)")
        pack_to_pretrain_dlc_model_name = gr.Textbox(label="模型名称")
        pack_to_pretrain_dlc_btn = gr.Button("打包为预训练 DLC")

        pack_to_pretrain_dlc_btn.click(
            self.pack_pretrain_dlc,
            inputs=[
                pack_to_pretrain_dlc_input_path,
                pack_to_pretrain_dlc_model_name,
            ],
        )

    def fn_pack_vocoder_dlc(self):
        gr.Markdown("## 打包为声码器 DLC")
        pack_to_vocoder_dlc_input_path = gr.Textbox(
            label="输入目录路径(可批量，取最后文件夹名作为模型名称)"
        )
        pack_to_vocoder_dlc_btn = gr.Button("打包为声码器 DLC")

        pack_to_vocoder_dlc_btn.click(
            self.pack_vocoder_dlc,
            inputs=[
                pack_to_vocoder_dlc_input_path,
            ],
        )

    def force_change_training_model_type(self, model_type_id):
        # 写入到 data/model_type
        with open("data/model_type", "w") as f:
            f.write(model_type_id)
        gr.Info("更改成功")

    def fn_force_change_training_model_type(self):
        gr.Markdown("## 强制更改模型类型")
        model_type_id = gr.Textbox(label="模型类型 ID")
        submit_btn = gr.Button("强制更改模型类型")

        submit_btn.click(self.force_change_training_model_type, inputs=[model_type_id])

    def __init__(self) -> None:
        self.fn_pack_pretrain_dlc()
        self.fn_force_change_training_model_type()
        self.fn_pack_vocoder_dlc()
