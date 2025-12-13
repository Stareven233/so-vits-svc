import sys
from functools import reduce

import librosa
import numpy as np
import torch
from torch.nn.modules.module import _addindent

from .constants import N_CLASS, CONST


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, "shape"):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        if file is sys.stdout:
            main_str += ", \033[92m{:,}\033[0m params".format(total_params)
        else:
            main_str += ", {:,} params".format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, "w")
        print(string, file=file)
        file.flush()

    return count


def to_local_average_cents(salience, center=None, thred=0.05):
    """
    优化版本：使用向量化操作和GPU加速的加权平均cents计算
    """
    # 确保是torch tensor
    if isinstance(salience, np.ndarray):
        salience = torch.from_numpy(salience)

    device = salience.device

    # 创建并缓存cents映射表
    if (
        not hasattr(to_local_average_cents, "cents_mapping")
        or to_local_average_cents.cents_mapping.device != device
    ):
        to_local_average_cents.cents_mapping = (
            20 * torch.arange(N_CLASS, device=device) + CONST
        )

    if salience.ndim == 1:
        if center is None:
            center = int(torch.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience_slice = salience[start:end]
        cents_slice = to_local_average_cents.cents_mapping[start:end]

        # 应用阈值
        if torch.max(salience_slice) <= thred:
            return 0

        product_sum = torch.sum(salience_slice * cents_slice)
        weight_sum = torch.sum(salience_slice)
        return (product_sum / weight_sum).item()

    if salience.ndim == 2:
        batch_size, n_class = salience.shape

        # 向量化处理批次
        if center is None:
            center = torch.argmax(salience, dim=1)

        # 创建索引张量
        indices = (
            torch.arange(n_class, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        center_expanded = center.unsqueeze(1)

        # 计算邻域掩码（向量化）
        start_idx = torch.clamp(center_expanded - 4, min=0)
        end_idx = torch.clamp(center_expanded + 5, max=n_class)

        mask = (indices >= start_idx) & (indices < end_idx)

        # 应用阈值掩码
        max_vals = torch.max(salience, dim=1, keepdim=True)[0]
        threshold_mask = max_vals > thred
        final_mask = mask & threshold_mask

        # 计算加权平均（向量化）
        cents_expanded = to_local_average_cents.cents_mapping.unsqueeze(0).expand(
            batch_size, -1
        )
        weighted_salience = salience * final_mask.float()

        numerator = torch.sum(weighted_salience * cents_expanded, dim=1)
        denominator = torch.sum(weighted_salience, dim=1)

        # 避免除零
        result = torch.zeros_like(numerator)
        valid_mask = denominator > 0
        result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

        return result

    raise Exception("label should be either 1d or 2d ndarray")


def to_viterbi_cents(salience, thred=0.05):
    """
    优化版本：缓存转移矩阵并使用更高效的viterbi解码
    """
    device = salience.device

    # 创建并缓存viterbi转移矩阵
    if (
        not hasattr(to_viterbi_cents, "transition")
        or to_viterbi_cents.transition.device != device
    ):
        xx, yy = torch.meshgrid(
            torch.arange(N_CLASS, device=device),
            torch.arange(N_CLASS, device=device),
            indexing="ij",
        )
        transition = torch.clamp(30 - torch.abs(xx - yy), min=0).float()
        transition = transition / transition.sum(dim=1, keepdim=True)
        to_viterbi_cents.transition = transition

    # 转换为概率
    prob = salience.T
    prob = prob / (prob.sum(dim=0, keepdim=True) + 1e-8)  # 添加小值避免除零

    # 执行viterbi解码
    try:
        # 将tensor移到CPU进行librosa处理
        prob_cpu = prob.detach().cpu().numpy()
        transition_cpu = to_viterbi_cents.transition.detach().cpu().numpy()
        path = librosa.sequence.viterbi(prob_cpu, transition_cpu).astype(np.int64)
        path = torch.from_numpy(path).to(device)
    except Exception:
        # 回退到简单的argmax
        path = torch.argmax(prob, dim=0)

    # 向量化计算local average cents
    batch_size = salience.shape[0]
    result = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        result[i] = to_local_average_cents(salience[i : i + 1], path[i : i + 1], thred)

    return result
