from easydict import EasyDict

def create_hparams():
    common_hparams = EasyDict(
                   # Data params
                   n_mels=80, 
                   n_symbols=256,
        
                   # Alignment params 
                   mean_coeff = 8,
                   scale_coeff = 8,
        
                   # Training Params
                   root_dir='/home/scpark/hard/datasets/LJSpeech-1.1/',
                   batch_size=32,
                   num_workers=1,
                   annealing_steps=50000,
                   lr=1e-4,
                   weight_decay=1e-6
                   )

    stt_hparams = EasyDict(
                   common_hparams,
                       
                   # STT Encoder params
                   embedding_dim = 128,
                   encoder_n_convs = 3,
                   encoding_dim = 128,
                   encoder_kernel_size = 5,
        
                   # STT Decoder params
                   prenet_dim = 128,
                   attention_rnn_dim = 256,
                   decoder_rnn_dim = 256,
                   p_attention_dropout = 0.1,
                   p_decoder_dropout = 0.1,
        
                  )
    
    tts_hparams = EasyDict(
                    common_hparams,
                    
                    # TTSTextEncoder params
                    text_encoder_n_convs = 3,
                    text_encoder_dim = 128,
                    text_encoder_kernel_size = 5,
                    embedding_dim = 128,
                    
                    # TTSMelEncoder&Decoder params
                    n_layers = 4,
                    n_blocks = 3,
                    enc_dim = 128,
                    enc_hidden_dim = 128,
                    dec_dim = 128,
                    dec_hidden_dim = 128,
                    z_dim = 16,
                    conv_type = 2,
                    encoder_residual = True,
                    decoder_residual = True,
                    decoder_expand_dim = False
                   )

    return stt_hparams, tts_hparams