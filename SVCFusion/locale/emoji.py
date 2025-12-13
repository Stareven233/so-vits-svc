from SVCFusion.locale.base import Locale

locale_name = "emoji"
locale_display_name = "😉✌"


class _Locale(Locale):
    unknown_model_type_tip = "❓🤖❗️🔍🛠️➡️🤖📊"
    preprocess_failed_tip = "⚠️🔄❌❗️📸🖥️📋➕👥💬"

    error_when_infer = "⚠️🧠❌<br>⏭️{1}📄<br>🔍🖥️<br>{2}"

    class device_chooser(Locale.device_chooser):
        device_dropdown_label = "💻"

    class model_chooser(Locale.model_chooser):
        submit_btn_value = "🔍🤖"
        model_type_dropdown_label = "🤖📊"
        search_path_label = "🔍📁"

        workdir_name = "📂💼"
        archive_dir_name = "📦🏋️‍♂️"
        models_dir_name = "📂🤖"

        no_model_value = "⛔🤖"
        unuse_value = "🚫"
        no_spk_value = "⛔🗣️"

        choose_model_dropdown_prefix = "🔍🤖"

        refresh_btn_value = "🔄"

        spk_dropdown_label = "🔍🗣️"
        no_spk_option = "🚫🤖"

    class form(Locale.form):
        submit_btn_value = "✅"
        audio_output_1 = "🔊"
        audio_output_2 = "🔊/🎵"
        audio_output_3 = "🔊/🎚️"
        textbox_output = "📝"

        dorpdown_liked_checkbox_yes = "✅"
        dorpdown_liked_checkbox_no = "❌"

        cancel_btn_value = "❌"
        canceling_tip = "⏳❌"

    class model_manager(Locale.model_manager):
        choose_model_title = "🔍🤖"
        action_title = "🎯"

        pack_btn_value = "📦🤖"
        pack_result_label = "📦📊"
        packing_tip = "⏳📦❗️🚫🔄👆"
        unpackable_tip = "🤖🚫📦"

        clean_log_btn_value = "🧹📋❗️✅🚫🏋️‍♂️➡️🧹"

        change_model_type_info = """
        #### 🔄🤖📊
        ❗️🔍❓🤖📊❗️🚫🔄🤖📊❗️🔄🔍🤖📊❗️
        """
        change_model_type_btn_value = "✅🔄"
        change_success_tip = "🔄✅"
        change_fail_tip = "🔄❌"

        move_folder_tip = "#### 📦➡️📂🤖"
        move_folder_name = "🤖📛"
        move_folder_name_auto_get = "🤖📛"
        move_folder_btn_value = "📦"
        other_text = "➕"
        moving_tip = "⏳📦❗️🚫🔄👆"
        moved_tip = "📦➡️{1}❗️🔄➡️✅"

    class main_ui(Locale.main_ui):
        release_memory_btn_value = "🧹💾/💻❗️⏏️🤖🔊"
        released_tip = "🧹💾/💻"
        infer_tab = "💡🧠"
        preprocess_tab = "⏳🔄"
        train_tab = "🏋️‍♂️"
        tools_tab = "🛠️"
        settings_tab = "⚙️"

        unloaded_model_tip = "🤖🚫📥❗️🔍📥🤖"

        model_tools_tab = "🤖🔧"
        audio_tools_tab = "🔊🔧"
        realtime_tools_tab = "⏱️"
        dlc_install_tools_tab = "📥"
        cuda_benchmark_tools_tab = "💻📊"

        start_cuda_benchmark_btn_value = "▶️💻📊"

        start_ddsp_realtime_gui_btn = "▶️🎛️6.0⏱️"
        start_ddsp6_1_realtime_gui_btn = "▶️🎛️6.1⏱️"

        starting_tip = "⏳▶️❗️🚫🔄👆❗️⚠️"

        load_model_btn_value = "📥🤖"
        infer_btn_value = "▶️🧠"

        model_manager_tab = "🤖📊"
        install_model_tab = "📥🤖"
        fish_audio_preprocess_tab = "🔊🔄"
        vocal_separation_tab = "🎤🔄"
        vocal_key_shift_tab = "🎤🔑"
        compatible_tab = "🤖🔄"

        move_folder_name_error = "🤖📛❌"

        detect_spk_tip = "🔍🗣️: "
        spk_not_found_tip = "🔍⛔🗣️"

    class DLC(Locale.DLC):
        dlc_install_label = "⬆️📥"
        dlc_install_btn_value = "📥"
        dlc_installing_tip = "⏳📥"
        dlc_install_success = "📥✅❗️🔄🌐➡️👀🆕"
        dlc_install_failed = "📥❌"
        dlc_install_empty = "⛔📄"
        dlc_install_ext_error = "🚫📄❌.sf_dlc"

    class compatible_models(Locale.compatible_models):
        upload_error = "⬆️❌❗️🔍📄✅"
        model_name_label = "🤖📛"
        upload_success = "⬆️✅"
        model_exists = "🤖✅"

        compatible_sovits = "🎤🤖🔄"
        sovits_main_model_label = "🎤🤖"
        sovits_diff_model_label = "🎤🔀"
        sovits_cluster_model_label = "🎤📊"

        sovits_main_model_config_label = "🎤🤖⚙️"
        sovits_diff_model_config_label = "🎤🔀⚙️"

    class preprocess(Locale.preprocess):
        tip = """
            📂🎵.wav➡️📁/👤

            👤1️⃣👤2️⃣👤3️⃣...

            📂➡️

            ```
            📁/
            |-👤1️⃣/
            |  | 1.wav
            |  | 2.wav
            |  | 3.wav
            |  ...
            |-👤2️⃣/
            |  | 1.wav
            |  | 2.wav
            |  | 3.wav
            |  ...
            ```

            ❓👇🔄
            
            ✅👇🔧🔄
            
            **💻➡️FCPE F0**
        """
        low_vram_tip = """
            ## 💻💾<6GB❗️➡️🎛️
            
            ⚠️ ❗️❗️🚫🏋️‍♂️❗️❗️
        """

        open_dataset_folder_btn_value = "📂📁"

        choose_model_label = "🔍🤖"
        start_preprocess_btn_value = "▶️🔄"

    class auto_mixing(Locale.auto_mixing):
        title = "🎵 🤖🎚️"
        description = "🤖🎚️➡️🎤🎵🤖🔧❗️👂🏽➡️🤖"
        support_info = "✅🎵🎤🤖🔧"
        upload_warning = "⚠️ ⚠️: ⬆️🎤🎵❗️🚫⬆️🎚️🔊"
        contributor_info = "🙏 @祡🍊 🤖🎚️🔧✅❗️"

        input_vocals_label = "📥🎤"
        input_bgm_label = "📥🎵"

        music_genre_label = "🎵🎨"
        music_genre_info = "🔍🎵🎨➡️👍🎚️"
        music_genre_pop = "🎵流行"
        music_genre_rock = "🎸🎵"
        music_genre_jazz = "🎺🎵"
        music_genre_electronic = "🔊🎵"
        music_genre_folk = "🎻🎵"
        music_genre_classical = "🎼🎵"

        voice_type_label = "🎤📊"
        voice_type_info = "🔍🎤📊➡️🔧"
        voice_type_male_low = "👨🎤(⬇️)"
        voice_type_male_high = "👨🎤(⬆️)"
        voice_type_female = "👩🎤"
        voice_type_rap = "🎤💬"
        voice_type_vocal = "🎤🎵"

        advanced_options_title = "⬆️"
        professional_params_title = "🤖⚙️🔧"

        deesser_strength_label = "⛔💥"
        deesser_strength_info = "⬇️🎤💥'S'"
        deesser_strength_off = "⛔"
        deesser_strength_light = "💡"
        deesser_strength_moderate = "🔧"
        deesser_strength_heavy = "💪"

        compression_strength_label = "📊💪"
        compression_strength_info = "🔧📊➡️🔊👂"
        compression_strength_light = "💡"
        compression_strength_moderate = "🔧"
        compression_strength_heavy = "💪"

        eq_style_label = "EQ🎨"
        eq_style_info = "🔊👂🎨🔧"
        eq_style_neutral = "🔧"
        eq_style_bright = "✨"
        eq_style_warm = "🔥"
        eq_style_vintage = "📺"

        reverb_level_label = "🔊📊"
        reverb_level_info = "🏠👂🔧"
        reverb_level_dry = "🏜️"
        reverb_level_subtle = "💡"
        reverb_level_light = "💡"
        reverb_level_moderate = "🔧"
        reverb_level_heavy = "💪"
        reverb_level_extreme = "🚀"

        echo_level_label = "📢📊"
        echo_level_info = "➕📢"
        echo_level_off = "⛔"
        echo_level_light = "💡"
        echo_level_moderate = "🔧"
        echo_level_heavy = "💪"

        submit_btn_value = "▶️🎚️"
        output_mixed_label = "🎚️📤"

        processing_start_log = "▶️🤖🎚️🔄..."
        processing_complete_log = "🤖🎚️🔄✅"

    class train(Locale.train):
        current_train_model_label = "🏋️‍♂️🤖"

        fouzu_tip = "~~🤖🛐➡️❓~~"

        gd_plus_1 = "👆➕🛐"
        gd_plus_1_tip = "🛐+1️⃣❗️💥-1️⃣"

        choose_sub_model_label = "🔍🤖"
        choose_pretrain_model_label = "🔍⏮️🤖"
        choose_pretrain_model_info = "🔍💻➡️⏮️🤖❗️🌐📱➡️⏮️🤖"
        pretrain_model_not_found_tip = "🔍⛔⏮️🤖"
        pretrain_model_not_found_suggestion = '<div style="background: var(--block-background-fill); padding: 8px;">🤖⛔⏮️🤖❗️📥 <a href="https://huggingface.co/collections/SVCFusion/pretrain-683b2fcfa4edab5a2942ba57" target="_blank">🌐</a> 📥</div>'

        pretrain_model_vec = "🎛️"
        pretrain_model_vocoder = "🔊"
        pretrain_model_size = "🕸️"
        pretrain_model_attn = "❓🔍"
        official_pretrain_model = "🏢⏮️🤖"

        load_pretrained_failed_tip = "📥⏮️🤖❌❗️🤖❌⚙️❓⛔⏮️🤖"

        start_train_btn_value = "▶️/➡️🏋️‍♂️"

        archive_btn_value = "📦💼"
        stop_btn_value = "⏹️🏋️‍♂️"

        archiving_tip = "⏳📦❗️🚫🔄👆"
        archived_tip = "📦✅❗️👀📂"

        stopped_tip = "⏹️🏋️‍♂️❗️👀🏋️‍♂️🖥️"

        tensorboard_btn = "▶️📊"

        launching_tb_tip = "⏳▶️📊❗️⏳"
        launched_tb_tip = "📊➡️{1}"

    class settings(Locale.settings):
        page = "📄"

        save_btn_value = "💾⚙️"

        pkg_settings_label = "📦⚙️"
        infer_settings_label = "🧠⚙️"
        sovits_settings_label = "🎤⚙️"
        ddsp6_settings_label = "🎛️6⚙️"
        ddsp6_1_settings_label = "🎛️6.1⚙️"

        class pkg(Locale.settings.pkg):
            lang_label = "🌐"
            lang_info = "🔄🌐➡️🔄📦"
            workdir_location_label = "📂💼"
            workdir_location_info = "⚙️📂💼📍📭⏮️exp/workdir"

        class infer(Locale.settings.infer):
            msst_device_label = "🔄💻"

        class sovits(Locale.settings.sovits):
            resolve_port_clash_label = "🔄🔌🚫❗️🪟✅"

        class ddsp6(Locale.settings.ddsp6):
            pretrained_model_preference_dropdown_label = "⏮️🔍"
            default_pretrained_model = "⏮️🤖 512 6"
            large_pretrained_model = "⬆️⏮️🤖 1024 12"

        class ddsp6_1(Locale.settings.ddsp6_1):
            pretrained_model_preference_dropdown_label = "⏮️🔍"
            default_pretrained_model = "⏮️(⬆️)🤖 1024 10"

        saved_tip = "💾✅"

    class install_model(Locale.install_model):
        tip = """
        ## ✅📥.sf_pkg/.h0_ddsp_pkg_model
        """

        file_label = "⬆️🤖📦"

        model_name_label = "🤖📛"
        model_name_placeholder = "✏️🤖📛"

        submit_btn_value = "📥🤖"

        installing_tip = "⏳📥🤖❗️⏳"
        install_success = "📥✅❗️🔄🌐👀🆕🤖"
        install_failed = "📥❌"
        file_empty = "⬆️🤖📦❓⏳⬆️"
        model_name_empty = "✏️🤖📛"
        unsupported_format = "🚫🤖📦"
        unsupported_model = "🤖🚫📥"

    class path_chooser(Locale.path_chooser):
        input_path_label = "📂📥"
        output_path_label = "📂📤"

    class fish_audio_preprocess(Locale.fish_audio_preprocess):
        to_wav_tab = "🔄WAV"
        slice_audio_tab = "✂️🔊"
        preprocess_tab = "🔄📊"
        max_duration_label = "⏱️⬆️"
        submit_btn_value = "▶️"

        input_output_same_tip = "📥📤📂✅"
        input_path_not_exist_tip = "📥📂⛔"

    class vocal_separation(Locale.vocal_separation):
        input_audio_label = "📥🔊"
        input_path_label = "📥📂"
        output_path_label = "📤📂"

        use_batch_label = "✅🔄"
        use_de_reverb_label = "⛔🔊"
        use_harmonic_remove_label = "⛔🎵"

        submit_btn_value = "▶️"
        vocal_label = "📤-🎤"
        inst_label = "📤-🎵"

        batch_output_message_label = "🔄📤💬"

        no_file_tip = "⛔📄"
        no_input_tip = "⛔📥📂"
        no_output_tip = "⛔📤📂"
        input_not_exist_tip = "📥📂⛔"
        output_not_exist_tip = "📤📂⛔"
        input_output_same_tip = "📥📤📂✅"

        finished = "✅"
        error_when_processing = "🔄❌❗️📸🖥️➡️❓"

        unusable_file_tip = "{1} ⏭️, 📄❌"

        batch_progress_desc = "📊"

        job_to_progress_desc = {
            "vocal": "⛔🎤",
            "kim_vocal": "⛔🎤",
            "deverb": "⛔🔊",
            "karaoke": "⛔🎵",
        }

    class common_infer(Locale.common_infer):
        audio_label = "🔊📄"

        use_batch_label = "✅🔄"

        precision_info = "🧠📊"
        precision_label = "📊"

        vocoder_label = "🔊"

        unknown_vocoder_tip = "❓🔊❗️🔍"
        vocoder_not_loaded_tip = "🔊🚫📥❗️🔍"

        use_vocal_separation_label = "⛔🎵"
        use_vocal_separation_info = "❓⛔🎵"

        use_de_reverb_label = "⛔🔊"
        use_de_reverb_info = "❓⛔🔊"

        use_harmonic_remove_label = "⛔🎵"
        use_harmonic_remove_info = "❓⛔🎵"

        use_automix_label = "🤖🎚️"
        use_automix_info = "❓🤖🎚️(✅⛔🎵➡️✅)"

        automix_but_not_vocal_separation_tip = "🤖🎚️➡️✅⛔🎵"

        f0_label = "f0 🔍"
        f0_info = "➡️🔍🤖"

        keychange_label = "🔑"
        keychange_info = "👨➡️👩 +12️⃣❗️👩➡️👨 -12️⃣❗️🔊❌🔍🔑"

        threshold_label = "✂️📊"
        threshold_info = "🎤✂️📊❗️🔊🔉➡️-40️⃣⬆️"

        vocal_register_shift_label = "🎤🔑"
        vocal_register_shift_info = "🎤🔑❗️🔊🔑➡️🎤🎵"
        vocal_register_shift_no_support_tip = "🔊🚫🎤🔑❗️🎤🔑⚙️➡️⛔"

    class ddsp_based_infer(Locale.ddsp_based_infer):
        method_label = "🧪"
        method_info = "➡️reflow🧪"

        infer_step_label = "🧠🔢"
        infer_step_info = "🧠🔢❗️✅"

        t_start_label = "⏱️▶️"
        t_start_info = "❓"

        num_formant_shift_key_label = "🔉🔑"
        num_formant_shift_key_info = "⬆️➡️🔊➕⬇️➡️🔊⬇️"

    class ddsp_based_preprocess(Locale.ddsp_based_preprocess):
        method_label = "🧪"
        method_info = "➡️reflow🧪"

    class common_preprocess(Locale.common_preprocess):
        encoder_label = "🔊🎛️"
        encoder_info = "➡️🔊🎛️🤖"

        slicer_num_workers_label = "✂️💻"
        slicer_num_workers_info = "✂️💻❗️💥💾➡️1️⃣"

        f0_label = "f0 🔍"
        f0_info = "➡️🔍🤖"

    class sovits(Locale.sovits):
        dataset_not_complete_tip = "📊❌❗️🔍📊❓🔄"
        finished = "✅"

        class train_main(Locale.sovits.train_main):
            log_interval_label = "📋⏱️"
            log_interval_info = "N➡️📋"

            eval_interval_label = "✅⏱️"
            eval_interval_info = "N➡️💾✅"

            all_in_mem_label = "💾📊"
            all_in_mem_info = "📊➡️💾➡️🏋️‍♂️❗️➕🏋️‍♂️🏃‍♂️❗️➡️💾"

            keep_ckpts_label = "💾💯"
            keep_ckpts_info = "💾N💯"

            batch_size_label = "🏋️‍♂️📊📏"
            batch_size_info = "⬆️➡️👍❗️⬆️➡️💾"

            learning_rate_label = "📚📈"
            learning_rate_info = "📚📈"

            num_workers_label = "📊🔄💻"
            num_workers_info = "✅💻>4❗️⬆️➡️👍"

            half_type_label = "📊"
            half_type_info = "🔍fp16➡️⏩❗️💥📈"

        class train_diff(Locale.sovits.train_diff):
            batchsize_label = "🏋️‍♂️📊📏"
            batchsize_info = "⬆️➡️👍❗️⬆️➡️💾❗️⚠️<📊"

            num_workers_label = "🏋️‍♂️💻"
            num_workers_info = "💻👍➡️0️⃣"

            amp_dtype_label = "🏋️‍♂️📊"
            amp_dtype_info = "🔍fp16❗️bf16➡️⏩❗️💥📈"

            lr_label = "📚📈"
            lr_info = "🚫🔄"

            interval_val_label = "✅⏱️"
            interval_val_info = "N➡️✅❗️💾"

            interval_log_label = "📋⏱️"
            interval_log_info = "N➡️📋"

            interval_force_save_label = "💪💾⏱️"
            interval_force_save_info = "N➡️💾"

            gamma_label = "📈⬇️"
            gamma_info = "🚫🔄"

            cache_device_label = "💾💻"
            cache_device_info = "🔍cuda➡️⏩❗️➡️💾(🎤🤖❌)"

            cache_all_data_label = "💾📊"
            cache_all_data_info = "➡️⏩❗️➡️💾"

            epochs_label = "🏋️‍♂️🔢"
            epochs_info = "📈➡️⏹️🏋️‍♂️"

            use_pretrain_label = "⏮️🤖"
            use_pretrain_info = "✅➡️⬇️🏋️‍♂️⏱️❗️❓❗️🚫🔄"

        class train_cluster(Locale.sovits.train_cluster):
            cluster_or_index_label = "📊❓🔍"
            cluster_or_index_info = "🏋️‍♂️📊❓🔍❓🔍➡️📊"

            use_gpu_label = "💻"
            use_gpu_info = "💻➡️⏩❗️⚙️📊✅"

        class infer(Locale.sovits.infer):
            cluster_infer_ratio_label = "📊/🔍📊"
            cluster_infer_ratio_info = "📊/🔍📊❗️0️⃣-1️⃣❗️❌🏋️‍♂️📊❓🔍➡️0️⃣"

            linear_gradient_info = "2️⃣🔊✂️🔀⏱️"
            linear_gradient_label = "🔀⏱️"

            k_step_label = "🔀🔢"
            k_step_info = "⬆️➡️🔀🤖❗️✅100"

            enhancer_adaptive_key_label = "🆙🔑"
            enhancer_adaptive_key_info = "🆙🔑(🔢)|✅0️⃣"

            f0_filter_threshold_label = "f0🔍📊"
            f0_filter_threshold_info = "✅crepe. 0️⃣-1️⃣. ⬇️📊➡️⬇️🎵⬆️❌"

            audio_predict_f0_label = "🤖f0🔍"
            audio_predict_f0_info = "🎤🔄🤖🎵❗️🔄🎵🚫✅➡️🎵⬆️"

            second_encoding_label = "2️⃣🎛️"
            second_encoding_info = "🔀⏮️➡️2️⃣🎛️❗️❓❗️⏱️👍❗️⏱️👎"

            clip_label = "💪✂️⏱️"
            clip_info = "💪🔊✂️⏱️, 0️⃣ ➡️🚫💪"

        class preprocess(Locale.sovits.preprocess):
            use_diff_label = "🏋️‍♂️🔀"
            use_diff_info = "✅➡️🏋️‍♂️🔀📄❗️➡️⏳"

            vol_aug_label = "🔊📥"
            vol_aug_info = "✅➡️🔊📥"

            num_workers_label = "💻"
            num_workers_info = "⬆️➡️⏩, ⬅️✅"

            subprocess_num_workers_label = "💻🧵"
            subprocess_num_workers_info = "⬆️➡️⏩, ⬅️✅"

            debug_label = "❓Debug"
            debug_info = "✅➡️🐞💬❗️❌❓🚫"

            use_old_preprocess_label = "⬅️🔄"
            use_old_preprocess_info = "⬅️🔄❗️🔄❌➡️"

        class model_types(Locale.sovits.model_types):
            main = "🤖"
            diff = "🔀"
            cluster = "📊/🔍🤖"

        class model_chooser_extra(Locale.sovits.model_chooser_extra):
            enhance_label = "NSFHifigan 🔊⬆️"
            enhance_info = "📊➕🤖👍❗️🏋️‍♂️👍🤖👎"

            feature_retrieval_label = "✅🔍📥"
            feature_retrieval_info = "❓🔍📥❗️✅📊🤖➡️❌"

            only_diffusion_label = "🔀"
            only_diffusion_info = "🧠🔀🤖❗️🚫"

    class ddsp6(Locale.ddsp6):
        infer_tip = "🧠🎛️🤖"

        class model_types(Locale.ddsp6.model_types):
            cascade = "📚🤖"

        class train(Locale.ddsp6.train):
            batch_size_label = "🏋️‍♂️📊📏"
            batch_size_info = "⬆️➡️👍❗️⬆️➡️💾❗️⚠️<📊"

            num_workers_label = "🏋️‍♂️💻"
            num_workers_info = "💻👍➡️0️⃣"

            amp_dtype_label = "🏋️‍♂️📊"
            amp_dtype_info = "🔍fp16❗️bf16➡️⏩❗️💥📈"

            lr_label = "📚📈"
            lr_info = "🚫🔄"

            interval_val_label = "✅⏱️"
            interval_val_info = "N➡️✅❗️💾"

            interval_log_label = "📋⏱️"
            interval_log_info = "N➡️📋"

            interval_force_save_label = "💪💾⏱️"
            interval_force_save_info = "N➡️💾"

            gamma_label = "📈⬇️"
            gamma_info = "🚫🔄"

            cache_device_label = "💾💻"
            cache_device_info = "🔍cuda➡️⏩❗️➡️💾(🎤🤖❌)"

            cache_all_data_label = "💾📊"
            cache_all_data_info = "➡️⏩❗️➡️💾"

            epochs_label = "🏋️‍♂️🔢"
            epochs_info = "📈➡️⏹️🏋️‍♂️"

            use_pretrain_label = "⏮️🤖"
            use_pretrain_info = "✅➡️⬇️🏋️‍♂️⏱️❗️❓❗️🚫🔄"

    class reflow(Locale.reflow):
        infer_tip = "🧠🔀🤖"

        class train(Locale.ddsp6.train):
            batch_size_label = "🏋️‍♂️📊📏"
            batch_size_info = "⬆️➡️👍❗️⬆️➡️💾❗️⚠️<📊"

            num_workers_label = "🏋️‍♂️💻"
            num_workers_info = "💻👍➡️0️⃣"

            amp_dtype_label = "🏋️‍♂️📊"
            amp_dtype_info = "🔍fp16❗️bf16➡️⏩❗️💥📈"

            lr_label = "📚📈"
            lr_info = "🚫🔄"

            interval_val_label = "✅⏱️"
            interval_val_info = "N➡️✅❗️💾"

            interval_log_label = "📋⏱️"
            interval_log_info = "N➡️📋"

            interval_force_save_label = "💪💾⏱️"
            interval_force_save_info = "N➡️💾"

            gamma_label = "📈⬇️"
            gamma_info = "🚫🔄"

            cache_device_label = "💾💻"
            cache_device_info = "🔍cuda➡️⏩❗️➡️💾(🎤🤖❌)"

            cache_all_data_label = "💾📊"
            cache_all_data_info = "➡️⏩❗️➡️💾"

            epochs_label = "🏋️‍♂️🔢"
            epochs_info = "📈➡️⏹️🏋️‍♂️"

            use_pretrain_label = "⏮️🤖"
            use_pretrain_info = "✅➡️⬇️🏋️‍♂️⏱️❗️❓❗️🚫🔄"

        class model_types(Locale.reflow.model_types):
            cascade = "📚🤖"

    class vocoder_key_shift(Locale.vocoder_key_shift):
        keyshift_slider_label = "🔑🔄"
        keyshift_slider_info = "🔢🔄🔊🎵"
        audio_input_label = "📥🔊"
        submit_btn_value = "🔄"
        output_audio_label = "📤🔊"
        vocoder_dropdown_label = "🔊🔍"
        update_vocoder_info = "🔍🔊"
        process_success_log = "🔊🔄✅❗️💾➡️{output_path}"
        process_failed_tip = "🔊🔄❌❗️🔍📥❓🔊⚙️"

    default_spk_name = "✅🗣️"

    preprocess_draw_desc = "📊✅"
    preprocess_desc = "🔄(📊➡️🖥️)"
    preprocess_finished = "🔄✅"
