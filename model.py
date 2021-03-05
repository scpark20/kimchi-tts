import torch
from torch import nn
from torch.nn import functional as F

from STT import STTModel
from TTS import TTSModel

class Model(nn.Module):
    def __init__(self, stt_hparams, tts_hparams):
        super().__init__()
        
        self.stt = STTModel(stt_hparams)
        self.tts = TTSModel(tts_hparams)
        
    def forward(self, batch, beta=1.0):
        stt_outputs = self.stt(batch)
        tts_outputs = self.tts(batch, stt_outputs, beta)
        
        return stt_outputs, tts_outputs
        
    def inference(self, cond, alignments=None, mel_length=None, temperature=1.0):
        y = self.tts.inference(cond, alignments, mel_length, temperature)
        
        return y