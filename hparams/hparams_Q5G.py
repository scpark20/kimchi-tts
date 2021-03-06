from easydict import EasyDict

def create_hparams():
    common_hparams = EasyDict(
                   # Data params
                   n_mels=80, 
                   n_symbols=256,
                   mel_norm=False,
                   dataset='lj',
                   data_dir='LJSpeech-1.1/',
                   data_file='metadata.csv',
                   g2p=True,
        
                   # Alignment params 
                   mean_coeff = 8,
                   scale_coeff = 8,
                   attention = 'Gaussian',
        
                   # Training Params
                   batch_size=16,
                   num_workers=1,
                   annealing_steps=50000,
                   lr=1e-4,
                   weight_decay=1e-6,
        
                   # Inference Params
                   truncated_min = -2,
                   truncated_max = 2, 
                   )

    stt_hparams = EasyDict(
                   common_hparams,
                       
                   # STT Encoder params
                   embedding_dim = 512,
                   encoder_n_convs = 3,
                   encoding_dim = 512,
                   encoder_kernel_size = 5,
        
                   # STT Decoder params
                   prenet_dim = 256,
                   attention_rnn_dim = 1024,
                   decoder_rnn_dim = 1024,
                   p_attention_dropout = 0.1,
                   p_decoder_dropout = 0.1,
        
                  )
    
    tts_hparams = EasyDict(
                    common_hparams,
                    
                    # TTSTextEncoder params
                    text_encoder_n_convs = 3,
                    text_encoder_dim = 512,
                    text_encoder_kernel_size = 5,
                    embedding_dim = 512,
                    
                    # TTSMelEncoder&Decoder params
                    n_layers = 5,
                    n_blocks = 3,
                    enc_dim = 256,
                    enc_hidden_dim = 256,
                    dec_dim = 256,
                    dec_hidden_dim = 256,
                    z_dim = 16,
                    conv_type = 8,
                    encoder_residual = True,
                    decoder_residual = True,
                    decoder_expand_dim = False,
                    z_proj = False,
                    enc_add = True
                   )

    return stt_hparams, tts_hparams