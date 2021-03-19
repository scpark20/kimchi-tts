from easydict import EasyDict

def create_hparams():
    common_hparams = EasyDict(
                   # Data params
                   n_mels=80, 
                   n_symbols=256,
                   mel_norm=False,
                   dataset='lj',
                   data_dir='/data/datasets/LJSpeech-1.1/',
                   data_file='metadata.csv',
                   g2p=True,
                   )

    stt_hparams = EasyDict(
                   common_hparams,
                   
                  )
    
    tts_hparams = EasyDict(
                    common_hparams,
                    
                   )

    return stt_hparams, tts_hparams