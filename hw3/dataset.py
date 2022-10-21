from typing import Dict, List
from torch import Tensor
from numpy import ndarray
from numpy.random.mtrand import RandomState
from pretty_midi.pretty_midi import PrettyMIDI

import os
import json
import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pretty_midi

from constants import HOP_SIZE, MAX_MIDI, MIN_MIDI, SAMPLE_RATE

def allocate_batch(batch:Dict[str,Tensor], device:torch.device) -> Dict[str,Tensor]:
    for key in batch.keys():
        if key != 'path':
            batch[key] = batch[key].to(device)
    return batch

class MAESTRO_small(Dataset):
    def __init__(self,
                 path:str = 'data',
                 groups:List[str] = None,
                 sequence_length:int = SAMPLE_RATE * 5,
                 hop_size:int = HOP_SIZE,
                 seed:int = 42,
                 random_sample:bool = True) -> None:
        self.path:str = path
        self.groups:list = groups if groups is not None else self.available_groups()
        assert all(group in self.available_groups() for group in self.groups)

        self.sample_length:int = ((sequence_length // hop_size) * hop_size) if sequence_length is not None else None
        self.random:RandomState = np.random.RandomState(seed)
        self.random_sample:bool = random_sample
        self.hop_size:int = hop_size

        self.data:List[Dict[str,Tensor]] = []

        print(f'Loading {len(groups)} group(s) of', self.__class__.__name__, 'at', path)
        for group in groups:
            file_list:List[tuple] = self.get_file_path_list_of_group(group)
            for input_files in tqdm(file_list, desc=f'Loading group {group}'):
                self.data.append(self.load(*input_files))

    @classmethod
    def available_groups(cls) -> List[str]:
        return ['train', 'validation', 'test', 'debug']

    def get_file_path_list_of_group(self, group:str) -> List[tuple]:
        metadata:List[dict] = json.load(open(os.path.join(self.path, 'data.json')))
        subset_name:str = 'train' if group == 'debug' else group

        files:List[tuple] = sorted([
                (os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                 os.path.join(self.path, row['midi_filename']))
                for row in metadata if row['split'] == subset_name
            ])

        if group == 'debug':
            files = files[:10]
        else:
            files = [(audio if os.path.exists(audio) else audio.replace(
                '.flac', '.wav'), midi) for audio, midi in files]

        return files
    
    def load(self, audio_path:str, midi_path:str) -> Dict[str,Tensor]:
        """Loads an audio track and the corresponding labels."""
        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE

        frames_per_sec:float = sr / self.hop_size

        audio_tensor:Tensor = torch.ShortTensor(audio)
        audio_length:int = len(audio_tensor)

        midi:PrettyMIDI = pretty_midi.PrettyMIDI(midi_path)
        midi_length_sec:float = midi.get_end_time()
        frame_length:int = min(int(midi_length_sec * frames_per_sec), (audio_length // self.hop_size) + 1)

        audio_tensor = audio_tensor[:frame_length * self.hop_size]

        frame:ndarray = midi.get_piano_roll(fs=frames_per_sec)

        onset = np.zeros_like(frame)
        for inst in midi.instruments:
            for note in inst.notes:
                onset[note.pitch, int(note.start * frames_per_sec)] = 1

        # to shape (time, pitch (88))
        frame_tensor:Tensor = torch.from_numpy(frame[MIN_MIDI:MAX_MIDI + 1].T)
        onset_tensor:Tensor = torch.from_numpy(onset[MIN_MIDI:MAX_MIDI + 1].T)
        data = dict(path=audio_path, audio=audio_tensor, frame=frame_tensor, onset=onset_tensor)
        return data
    
    def __getitem__(self, index:int):
        data:Dict[str,Tensor] = self.data[index]

        audio:Tensor = data['audio']
        frames:Tensor = (data['frame'] >= 1)
        onsets:Tensor = (data['onset'] >= 1)

        frame_len:int = frames.shape[0]

        if self.sample_length is not None:
            n_steps:int = self.sample_length // self.hop_size

            step_begin:int = self.random.randint(frame_len - n_steps) if self.random_sample else 0
            step_end:int = step_begin + n_steps
            
            sample_begin:int = step_begin * self.hop_size
            sample_end:int = sample_begin + self.sample_length

            audio_seg:Tensor = audio[sample_begin:sample_end]
            frame_seg:Tensor = frames[step_begin:step_end]
            onset_seg:Tensor = onsets[step_begin:step_end]

            result = dict(path=data['path'])
            result['audio'] = audio_seg.float().div_(32768.0)
            result['frame'] = frame_seg.float()
            result['onset'] = onset_seg.float()
        else:
            result = dict(path=data['path'])
            result['audio'] = audio.float().div_(32768.0)
            result['frame'] = frames.float()
            result['onset'] = onsets.float()

        return result

    def __len__(self) -> int:
        return len(self.data)