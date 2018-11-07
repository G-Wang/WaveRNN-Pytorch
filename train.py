"""Training WaveRNN Model.

usage: train.py [options] <data-root>

options:
    --checkpoint-dir=<dir>      Directory where to save model checkpoints [default: checkpoints].
    --checkpoint=<path>         Restore model from checkpoint path if given.
    -h, --help                  Show this help message and exit
"""
from docopt import docopt

import os
from os.path import dirname, join, expanduser
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import librosa

from model import build_model

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from model import build_model
from distributions import *
from loss_function import nll_loss
from dataset import raw_collate, discrete_collate, AudiobookDataset
from hparams import hparams as hp
from lrschedule import noam_learning_rate_decay, step_learning_rate_decay

global_step = 0
global_epoch = 0
global_test_step = 0
use_cuda = torch.cuda.is_available()

def save_checkpoint(device, model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(step))
    optimizer_state = optimizer.state_dict()
    global global_test_step
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch
    global global_test_step

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    global_test_step = checkpoint.get("global_test_step", 0)

    return model


def test_save_checkpoint():
    checkpoint_path = "checkpoints/"
    device = torch.device("cuda" if use_cuda else "cpu")
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    global global_step, global_epoch, global_test_step
    save_checkpoint(device, model, optimizer, global_step, checkpoint_path, global_epoch)

    model = load_checkpoint(checkpoint_path+"checkpoint_step000000000.pth", model, optimizer, False)


def evaluate_model(model, data_loader, checkpoint_dir, limit_eval_to=5):
    """evaluate model and save generated wav and plot

    """
    test_path = data_loader.dataset.test_path
    test_files = os.listdir(test_path)
    counter = 0
    output_dir = os.path.join(checkpoint_dir,'eval')
    for f in test_files:
        if f[-7:] == "mel.npy":
            mel = np.load(os.path.join(test_path,f))
            wav = model.generate(mel)
            # save wav
            wav_path = os.path.join(output_dir,"checkpoint_step{:09d}_wav_{}.wav".format(global_step,counter))
            librosa.output.write_wav(wav_path, wav, sr=hp.sample_rate)
            # save wav plot
            fig_path = os.path.join(output_dir,"checkpoint_step{:09d}_wav_{}.png".format(global_step,counter))
            fig = plt.plot(wav.reshape(-1))
            plt.savefig(fig_path)
            # clear fig to drawing to the same plot
            plt.clf()
            counter += 1
        # stop evaluation early via limit_eval_to
        if counter >= limit_eval_to:
            break


def train_loop(device, model, data_loader, optimizer, checkpoint_dir):
    """Main training loop.

    """
    # create loss and put on device
    if hp.input_type == 'raw':
        if hp.distribution == 'beta':
            criterion = beta_mle_loss
        elif hp.distribution == 'gaussian':
            criterion = gaussian_loss
    elif hp.input_type == 'mixture':
        criterion = discretized_mix_logistic_loss
    elif hp.input_type in ["bits", "mulaw"]:
        criterion = nll_loss
    else:
        raise ValueError("input_type:{} not supported".format(hp.input_type))

    

    global global_step, global_epoch, global_test_step
    while global_epoch < hp.nepochs:
        running_loss = 0
        for i, (x, m, y) in enumerate(tqdm(data_loader)):
            x, m, y = x.to(device), m.to(device), y.to(device)
            y_hat = model(x, m)
            y = y.unsqueeze(-1)
            loss = criterion(y_hat, y)
            # calculate learning rate and update learning rate
            if hp.fix_learning_rate:
                current_lr = hp.fix_learning_rate
            elif hp.lr_schedule_type == 'step':
                current_lr = step_learning_rate_decay(hp.initial_learning_rate, global_step, hp.step_gamma, hp.lr_step_interval)
            else:
                current_lr = noam_learning_rate_decay(hp.initial_learning_rate, global_step, hp.noam_warm_up_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            optimizer.zero_grad()
            loss.backward()
            # clip gradient norm
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_norm)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i+1)
            # saving checkpoint if needed
            if global_step != 0 and global_step % hp.save_every_step == 0:
                save_checkpoint(device, model, optimizer, global_step, checkpoint_dir, global_epoch)
            # evaluate model if needed
            if global_step != 0 and global_test_step !=True and global_step % hp.evaluate_every_step == 0:
                print("step {}, evaluating model: generating wav from mel...".format(global_step))
                evaluate_model(model, data_loader, checkpoint_dir)
                print("evaluation finished, resuming training...")

            # reset global_test_step status after evaluation
            if global_test_step is True:
                global_test_step = False
            global_step += 1
        
        print("epoch:{}, running loss:{}, average loss:{}, current lr:{}".format(global_epoch, running_loss, avg_loss, current_lr))
        global_epoch += 1



if __name__=="__main__":
    args = docopt(__doc__)
    #print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    data_root = args["<data-root>"]

    # make dirs, load dataloader and set up device
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir,'eval'), exist_ok=True)
    dataset = AudiobookDataset(data_root)
    if hp.input_type == 'raw':
        collate_fn = raw_collate
    elif hp.input_type == 'mixture':
        collate_fn = raw_collate
    elif hp.input_type in ['bits', 'mulaw']:
        collate_fn = discrete_collate
    else:
        raise ValueError("input_type:{} not supported".format(hp.input_type))
    data_loader = DataLoader(dataset, collate_fn=collate_fn, shuffle=True, num_workers=0, batch_size=hp.batch_size)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device:{}".format(device))

    # build model, create optimizer
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=hp.initial_learning_rate, betas=(
        hp.adam_beta1, hp.adam_beta2),
        eps=hp.adam_eps, weight_decay=hp.weight_decay,
        amsgrad=hp.amsgrad)

    if hp.fix_learning_rate:
        print("using fixed learning rate of :{}".format(hp.fix_learning_rate))
    elif hp.lr_schedule_type == 'step':
        print("using exponential learning rate decay")
    elif hp.lr_schedule_type == 'noam':
        print("using noam learning rate decay")

    # load checkpoint
    if checkpoint_path is None:
        print("no checkpoint specified as --checkpoint argument, creating new model...")
    else:
        model = load_checkpoint(checkpoint_path, model, optimizer, False)
        print("loading model from checkpoint:{}".format(checkpoint_path))
        # set global_test_step to True so we don't evaluate right when we load in the model
        global_test_step = True

    # main train loop
    try:
        train_loop(device, model, data_loader, optimizer, checkpoint_dir)
    except KeyboardInterrupt:
        print("Interrupted!")
        pass
    finally:
        print("saving model....")
        save_checkpoint(device, model, optimizer, global_step, checkpoint_dir, global_epoch)
    

def test_eval():
    data_root = "data_dir"
    dataset = AudiobookDataset(data_root)
    if hp.input_type == 'raw':
        collate_fn = raw_collate
    elif hp.input_type == 'bits':
        collate_fn = discrete_collate
    else:
        raise ValueError("input_type:{} not supported".format(hp.input_type))
    data_loader = DataLoader(dataset, collate_fn=collate_fn, shuffle=True, num_workers=0, batch_size=hp.batch_size)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device:{}".format(device))

    # build model, create optimizer
    model = build_model().to(device)

    evaluate_model(model, data_loader)

    

    

