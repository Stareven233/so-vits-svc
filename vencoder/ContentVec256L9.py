import torch
import torch.nn as nn
from transformers import HubertModel, HubertConfig

from util import logger
from vencoder.encoder import SpeechEncoder


class HubertModelWithFinalProj(HubertModel):
    '''带 final_proj 的 HuBERT 模型'''
    def __init__(self, config):
        super().__init__(config)
        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)
    
    def forward(self, input_values, output_layer=None):
        # output_layer: 指定返回哪一层 (1-12)，None 则返回最后一层
        outputs = super().forward(
            input_values,
            output_hidden_states=True,
            return_dict=True
        )
        
        if output_layer is not None:
            # hidden_states[0] 是 embedding 层，[1] 是第1层，[9] 是第9层
            hidden = outputs.hidden_states[output_layer]
        else:
            hidden = outputs.last_hidden_state
            
        return {
            "last_hidden_state": hidden,
            "projected_features": self.final_proj(hidden)
        }


class ContentVec256L9(SpeechEncoder):
    def __init__(
        self,
        vec_path='pretrain/contentvec/lengyue233_cvec-best.bin',
        device=None,
        log=True,
    ):

        if log:
            logger.info('load model(s) from {}'.format(vec_path))
        self.hidden_dim = 256
        self.dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.config = HubertConfig(
            classifier_proj_size=256,  # final_proj 输出维度
        )
        self.model = HubertModelWithFinalProj(self.config)
        state_dict  = torch.load(vec_path, map_location=self.dev)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.dev)
        self.model.eval()

    @torch.no_grad()
    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        
        # output_layer=9 对应第9层 transformer 输出
        outputs = self.model(feats.to(self.dev), output_layer=9)
        # 使用 final_proj 投影后的特征 (1, T, 256)
        feats_256 = outputs['projected_features']
        # 转置为 (1, 256, T) 以匹配原输出格式
        return feats_256.transpose(1, 2)
