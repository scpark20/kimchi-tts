from easydict import EasyDict

def create_hparams():
    common_hparams = EasyDict(
                   # Data params
                   n_mels=80, 
                   n_symbols=256,
        
                   # Alignment params 
                   mean_coeff = 8,
                   scale_coeff = 8,
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
                    text_encoder_dim = 128,
                    text_encoder_kernel_size = 5,
                    embedding_dim = 128,
                    
                    # TTSMelEncoder&Decoder params
                    n_layers = 5,
                    n_blocks = 3,
                    enc_dim = 128,
                    enc_hidden_dim = 256,
                    dec_dim = 128,
                    dec_hidden_dim = 256,
                    z_dim = 16,
                   )

    return stt_hparams, tts_hparams