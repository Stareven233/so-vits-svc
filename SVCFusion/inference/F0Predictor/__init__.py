import torch


f0_predictor_object = None
using_f0_predictor_name = None


def get_f0_predictor(f0_predictor, hop_length, sampling_rate, **kargs):
    global f0_predictor_object, using_f0_predictor_name
    if f0_predictor_object and using_f0_predictor_name == f0_predictor:
        return f0_predictor_object

    if f0_predictor == "pm":
        from modules.F0Predictor.PMF0Predictor import PMF0Predictor

        f0_predictor_object = PMF0Predictor(
            hop_length=hop_length, sampling_rate=sampling_rate
        )
    elif f0_predictor == "crepe":
        from modules.F0Predictor.CrepeF0Predictor import CrepeF0Predictor

        f0_predictor_object = CrepeF0Predictor(
            hop_length=hop_length,
            sampling_rate=sampling_rate,
            device=kargs["device"],
            threshold=kargs["threshold"],
        )
    elif f0_predictor == "harvest":
        from modules.F0Predictor.HarvestF0Predictor import HarvestF0Predictor

        f0_predictor_object = HarvestF0Predictor(
            hop_length=hop_length, sampling_rate=sampling_rate
        )
    elif f0_predictor == "dio":
        from modules.F0Predictor.DioF0Predictor import DioF0Predictor

        f0_predictor_object = DioF0Predictor(
            hop_length=hop_length, sampling_rate=sampling_rate
        )
    elif f0_predictor == "rmvpe":
        from modules.F0Predictor.RMVPEF0Predictor import RMVPEF0Predictor

        f0_predictor_object = RMVPEF0Predictor(
            hop_length=hop_length,
            sampling_rate=sampling_rate,
            dtype=torch.float32,
            device=kargs["device"],
            threshold=kargs["threshold"],
        )
    elif f0_predictor == "fcpe":
        from modules.F0Predictor.FCPEF0Predictor import FCPEF0Predictor

        f0_predictor_object = FCPEF0Predictor(
            hop_length=hop_length,
            sampling_rate=sampling_rate,
            dtype=torch.float32,
            device=kargs["device"],
            threshold=kargs["threshold"],
        )
    else:
        raise Exception("Unknown f0 predictor")

    using_f0_predictor_name = f0_predictor
    return f0_predictor_object
