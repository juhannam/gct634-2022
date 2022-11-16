import pickle
import os
import sys
from progress.bar import Bar
import utils
from midi_neural_processor.processor import encode_midi
from tqdm.contrib.concurrent import process_map

MAX_WORKERS = 12
MAX_SEQ = 2048 # Should set same as max_seq in config file. (ex) base.yml)

def preprocess_midi(vars):
    path, save_dir  = vars
    data = encode_midi(path)
    if len(data) >= 2048:
        with open('{}/{}.pickle'.format(save_dir, path.split('/')[-1]), 'wb') as f:
            pickle.dump(data, f)
    else:
        print(f'Do not save pickle because data is too short, shorter than {MAX_SEQ}')


def preprocess_midi_files_under(midi_root, save_dir):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    midi_paths_ = [(x, save_dir) for x in midi_paths]
    os.makedirs(save_dir, exist_ok=True)
    process_map(preprocess_midi, midi_paths_, max_workers=MAX_WORKERS, chunksize=1, desc=f"Preprocessing with {MAX_WORKERS} workers", colour='green')


if __name__ == '__main__':
    preprocess_midi_files_under(
            midi_root=sys.argv[1],
            save_dir=sys.argv[2])
