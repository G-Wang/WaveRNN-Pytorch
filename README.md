# WaveRNN-Pytorch
This repository contains Fatcord's [Alternative](https://github.com/fatchord/WaveRNN) WaveRNN (Faster training), which contains a fast-training, small GPU memory implementation of WaveRNN vocoder.

This repo refracts the code and adds slight modifications, and removes running on Jupyter notebook.
# Highlights
* support raw audio wav modelling (via a single Beta Distribution)
* relatively fast synthesis speed without much optimization yet (around 2000 samples/sec on GTX 1060 Ti, 16 GB ram, i5 processor)
* support Fatcord's original quantized (9-bit) wav modelling

# Audio Samples
1. [Single beta distribution](https://soundcloud.com/gary-wang-23/sets/wavernn-samples) on held-out testing data from LjSpeech. This is trained with the single Beta distribution.

2. [9-bit audio](https://soundcloud.com/gary-wang-23/sets/wave_rnn_9_bit_11k_step) on held-out testing data from LJSpeech. This model trains the fastest (this is around 130 epochs)

3. [10-bit audio](https://soundcloud.com/gary-wang-23/sets/wavernn-pytorch-10-bit-raw-audio-200k) on held-out testing data from LJSpeech. This model sounds and trains pretty close to 9 bit. We want the higher bit the better.

# Pretrained Checkpoints
1. [Single Beta Distribution](https://drive.google.com/open?id=138i0MtEkDqLM6fmBniQloEMtMlCHgJha) trained for 112k. Make sure to change `hparams.input_type` to `raw`.
2. [9-bit quantized audio](https://drive.google.com/open?id=114Xk3P9dD-_e2W8jmiKSpOX1UGb7qem3) trained for 11k, or around 130 epochs, can be trained further. Make sure to change `hparams.input_type` to `bits`.
3. [10-bit quantized audio](https://drive.google.com/open?id=1djWm62tHIndopyS5spkHf68lI6-h5a3H). To ensure your model is built properly, download the `hparams.py` [here](https://drive.google.com/open?id=1nXSW4u01bEbUkRW4Vd3IQ6soBAXPg6aw), either replace this with your local `hparams.py` file or note and update any changes.




# Requirements
* Python 3
* CUDA >=8.0
* PyTorch >= v0.4.1

# Installation
Ensure above requirements are met.

```
git clone https://github.com/G-Wang/WaveRNN-Pytorch.git
cd WaveRNN-Pytorch
pip install -r requirements.txt
```

# Usage
## 1. Adjusting Hyperparameters
Before running scripts, one can adjust hyperparameters in `hparams.py`.

Some hyperparameters that you might want to adjust:
* `input_type` (best performing ones are currently `bits` and `raw`, see `hparams.py` for more details)
* `batch_size`
* `save_every_step` (checkpoint saving frequency)
* `evaluate_every_step` (evaluation frequency)
* `seq_len_factor` (sequence length of training audio, the longer the more GPU it takes)
## 2. Preprocessing
This function processes raw wav files into corresponding mel-spectrogram and wav files according to the audio processing hyperparameters.

Example usage:
```
python preprocess.py /path/to/my/wav/files
```
This will process all the `.wav` files in the folder `/path/to/my/wav/files` and save them in the default local directory called `data_dir`.

Can include `--output_dir` to specify a specific directory to store the processed outputs.

## 3. Training
Start training process. checkpoints are by default stored in the local directory `checkpoints`.
The script will automatically save a checkpoint when terminated by `crtl + c`.


Example 1: starting a new model for training
```
python train.py data_dir
```
`data_dir` is the directory containing the processed files.

Example 2: Restoring training from checkpoint
```
python train.py data_dir --checkpoint=checkpoints/checkpoint0010000.pth
```
Evaluation `.wav` files and plots are saved in `checkpoints/eval`.

# WIP
- [ ] optimize learning rate schedule
- [ ] optimize training hyperparameters (seq_len and batch_size)
- [ ] batch generation for synthesis speedup
- [ ] model pruning








