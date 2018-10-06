class hparams:

    # option parameters

    # Input type:
    # 1. raw [-1, 1]
    # 2. mulaw-quantize [0, 256]
    # 3. bits [0, 512]
    #
    # If input_type is raw, network assumes scalar input and output scalar value sampled
    # from a single Beta distribution, otherwise one-hot input and softmax outputs are asumed.
    input_type = 'raw'
    #
    # distribution type, currently supports only 'beta'
    distribution = 'beta'
    #
    # mu_law dimension can be changed, but make sure input_type is mulaw-quantize
    mu = 256
    #
    # for Fatcord's original 9 bit audio, specify the audio bit rate. Note this corresponds to network output
    # of size 2**bits, so 9 bits would be 512 output, etc.
    bits = 9
    # note: r9r9's deepvoice3 preprocessing is used instead of Fatcord's original.
    #     
    # audio processing parameters
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
    

    # training parameters
    batch_size = 16
    # note the rnn's don't train too well with very long seq_len
    # it's recommended to keep them no longer than 6
    seq_len = 5
    epochs = 5000
    learning_rate = 1e-4