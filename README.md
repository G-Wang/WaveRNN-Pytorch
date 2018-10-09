# WaveRNN-Pytorch
This repository contains Fatcord's [Alternative](https://github.com/fatchord/WaveRNN) WaveRNN (Faster training), which contains a fast-training, small GPU memory implementation of WaveRNN vocoder.

This repo refracts the code and adds slight modifications, and removes running on Jupyter notebook.
# Highlights
* support raw audio wav modelling (via a single Beta Distribution)
* relatively fast synthesis speed without much optimization yet (around 2000 samples/sec on GTX 1060 Ti, 16 GB ram, i5 processor)
* support Fatcord's original quantized (9-bit) wav modelling

# Audio Samples
Some [audio samples](https://soundcloud.com/gary-wang-23/sets/wavernn-samples) on held-out testing data from LjSpeech. This is trained with the single Beta distribution.

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
## 1. Adusting Hyperparameters
Before running scripts, one can adjust hyperparameters in `hparams.py`.

Some hyperparameters that you might want to adjust:
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








