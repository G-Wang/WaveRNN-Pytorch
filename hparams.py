class hparams:

    # option parameters

    # Input type:
    # 1. raw [-1, 1]
    # 2. bits [0, 512]
    #
    # If input_type is raw, network assumes scalar input and output scalar value sampled
    # from a single Beta distribution, otherwise one-hot input and softmax outputs are asumed.
    input_type = 'raw'
    #
    # distribution type, currently supports only 'beta'
    distribution = 'beta'
    #
    # for Fatcord's original 9 bit audio, specify the audio bit rate. Note this corresponds to network output
    # of size 2**bits, so 9 bits would be 512 output, etc.
    bits = 9
    # note: r9r9's deepvoice3 preprocessing is used instead of Fatcord's original.
    #--------------     
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
    #----------------
    #
    #----------------
    # model parameters
    rnn_dims = 512
    fc_dims = 512
    pad = 2
    # note upsample factors must multiply out to be equal to hop_size, so adjust
    # if necessary (i.e 4 x 4 x 16 = 256)
    upsample_factors = (4, 4, 16)
    compute_dims = 128
    res_out_dims = 128
    res_blocks = 10
    #----------------
    #
    #----------------
    # training parameters
    batch_size = 16
    nepochs = 5000
    save_every_step = 5000
    evaluate_every_step = 5000
    # seq_len_factor can be adjusted to increase training sequence length (will increase GPU usage)
    seq_len_factor = 5
    seq_len = seq_len_factor * hop_size
    batch_size = 16
    grad_norm = 1.0
    #learning rate parameters
    initial_learning_rate=1e-3
    noam_warm_up_steps = 2000 * (batch_size // 16)
    adam_beta1=0.9
    adam_beta2=0.999
    adam_eps=1e-8
    amsgrad=False
    weight_decay = 0.0
    fix_learning_rate = None # modify if one wants to use a fixed learning rate
    #-----------------
