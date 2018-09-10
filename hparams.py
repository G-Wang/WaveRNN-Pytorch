class hparams:

    # option parameters
    # use mu-law encoding for audio. Note mu-law encodes audio into 256 logits
    # if not mu-law, the specified bit rate will be used in the audio parameter section
    use_mu_law = True # use mu-law encoding for audio

    # note r9r9's deepvoice3 preprocessing uses a slightly different preprocessing
    # from Fatcord's version.
    use_r9r9_deepvoice = True 
    
    # audio parameters
    num_mels = 80
    fmin = 125
    fmax = 7600
    fft_size = 1024
    hop_size = 256
    win_length = 1024
    sample_rate = 22050
    preemphasis = 0.97
    min_level_db = -100
    ref_level_db = 20
    rescaling = False
    rescaling_max = 0.999
    allow_clipping_in_normalization = True
    bits = 9

    # training parameters
    batch_size = 16
    # note the rnn's don't train too well with very long seq_len
    # it's recommended to keep them no longer than 6
    seq_len = 5
    epochs = 5000
    learning_rate = 1e-4