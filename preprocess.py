"""
Preprocess dataset

usage: preproess.py [options] <wav-dir> <output-dir>

options:
    -h, --help      Show help message.
"""
from docopt import docopt
import numpy as np
import math, pickle, os
from audio import *
from hparams import hparams as hp
from nnmnkwii import preprocessing as P
from utils import *
from tqdm import tqdm

def get_wav_mel(path):
    """Given path to .wav file, get the quantized wav and mel spectrogram as numpy vectors

    """
    wav = load_wav(path)
    mel = melspectrogram(wav)
    if hp.use_mu_law:
        quant = P.mulaw_quantize(wav)
    else:
        quant = (wav + 1.) * (2**hp.bits - 1) / 2
    return quant, mel

def process_data(wav_dir, output_dir):
    """given wav directory and output directory, process wav files and save quantized wav and mel
    spectrogram to output directory

    """
    dataset_ids = []
    # get list of wav files
    wav_files = os.listdir(wav_dir)
    for i, wav_file in enumerate(tqdm(wav_files)):
        # get the file id
        file_id = '{:d}'.format(i).zfill(5)
        quant, mel = get_wav_mel(wav_dir + wav_file)
        # save
        np.save(output_dir+"mel/"+file_id+".npy", mel)
        np.save(output_dir+"quant/"+file_id+".npy", quant)
        # add to dataset_ids
        dataset_ids.append(file_id)

    # save dataset_ids
    with open(output_dir + 'dataset_ids.pkl', 'wb') as f:
        pickle.dump(dataset_ids, f)

    print("\npreprocessing done, total processed wav files:{}." 
    "\nProcessed files are located in:{}".format(len(wav_files), os.path.abspath(output_dir)))



if __name__=="__main__":
    args = docopt(__doc__)
    wav_dir = args["<wav-dir>"]
    output_dir = args["<output-dir>"]

    # check path name
    wav_dir = check_path_name(wav_dir)
    output_dir = check_path_name(output_dir)

    # check if output_dir exits, if not create
    if not os.path.exists(output_dir):
        print("\noutput path does not exit, creating....")
        os.mkdir(output_dir)
        os.mkdir(output_dir+"mel/")
        os.mkdir(output_dir+"quant/")

    if not os.path.exists(output_dir+"mel/"):
        os.mkdir(output_dir+"mel/")
        
    if not os.path.exists(output_dir+"quant/"):
        os.mkdir(output_dir+"quant/")

    # process data
    process_data(wav_dir, output_dir)