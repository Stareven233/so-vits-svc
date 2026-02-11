import torch
from transformers import HubertConfig

from util import logger
from vencoder.encoder import SpeechEncoder
from vencoder.ContentVec256L9 import HubertModelWithFinalProj


class ContentVec768L12(SpeechEncoder):
    def __init__(
        self,
        vec_path="pretrain/contentvec/lengyue233_cvec-best.bin",
        device=None,
        log=True,
    ):
        super().__init__()

        if log:
            logger.info('load model(s) from {}'.format(vec_path))
        self.hidden_dim = 768
        self.dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 直接使用 HubertModelWithFinalProj，假设权重已转换
        self.model = HubertModelWithFinalProj(HubertConfig())
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

        outputs = self.model(feats.to(self.dev))
        logits = outputs['last_hidden_state']  # 或根据实际输出调整
        return logits.transpose(1, 2)
