"""
Preprocess dataset

usage: preproess.py [options] <wav-dir> <output-dir>

options:
    -h, --help      Show help message.
"""
from docopt import docopt
import numpy as np
import math, pickle, os
from deepvoice_audio import *
from hparams import hparams as hp
from tqdm import tqdm

def get_wav_mel(path):
    """Given path to .wav file, get the quantized wav and mel spectrogram as numpy vectors

    """
    wav = load_wav(path)
    mel = melspectrogram(wav)
    quant = (wav + 1.) * (2**hp.bits - 1) / 2
    return quant, mel

def process_data(wav_dir, output_dir):
    """given wav directory and output directory, process wav files and save quantized wav and mel
    spectrogram to output directory

    """
    dataset_ids = []
    # get list of wav files
    wav_files = os.listdir(wav_dir)
    for wav_file in tqdm(wav_files):
        # get the file id
        file_id = wav_file.split('/')[-1][:-4] # skip the .wav
        #quant, mel = get_wav_mel(wav_dir + wav_file)
        # save
        #np.save(output_dir+"/me/"+file_id+".npy", mel)
        #np.save(output_dir+"/quant/"+file_id+".npy", quant)
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
    if wav_dir[-1] != "/":
        wav_dir += "/"
    if output_dir[-1] != "/":
        output_dir += "/"

    # check if output_dir exits, if not create
    if not os.path.exists(output_dir):
        print("\noutput path does not exit, creating....")
        os.mkdir(output_dir)
        os.mkdir(output_dir+"mel/")
        os.mkdir(output_dir+"quant/")

    # process data
    process_data(wav_dir, output_dir)