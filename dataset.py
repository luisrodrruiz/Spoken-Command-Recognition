import os
import torch
import torchaudio
import torchaudio.transforms as T
import sys
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence



def collate_fn(batch):
    """
    Pads batch of variable length sequences.

    Args:
      batch: A list of tuples (sequence, label).

    Returns:
      padded_sequences: A tensor of padded sequences.
      targets: A tensor of targets.
      lengths: A tensor of sequence lengths.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True) # padding at the end of the sequence
    labels = torch.tensor(labels)
    return padded_sequences, labels, lengths

# Dataset for  fashion_mnist 
class AudioDataset(Dataset):
    """ Audio dataset
        The dataset is expected to be in a csv file, where each row will
        contain at least two columns: filename and label.
    """
    def __init__(self, datafile,  audio_path = ''):
        """ The constructor loads the dataset into memory
            Parameters:
               datafile: csv file containing the dataset
               audio_path: base path for the audio files. If not specified
                            it will simply use the paths in the csv files. 
                            The wavefiles are expected to be sampled at 16Khz
                            and 16 bits per sample. 
        """

        mel_spectrogram = T.MelSpectrogram(
            n_fft=512,
            n_mels=64,
            mel_scale="htk",
        )

        
        self.features = []
        self.labels = []
        tmp_label_map = {}
        df = pd.read_csv(datafile)
        for index, row in df.iterrows():
            label = row['label']
            if not label in tmp_label_map:
                tmp_label_map[label] = len(tmp_label_map)
            self.labels.append(tmp_label_map[label])
            filename = row['filename']

            waveform, sample_rate = torchaudio.load(os.path.join(audio_path,filename))
            if sample_rate != 16000:
                print('WARNING !!! Sample rate for file ', filename, ' is ', sample_rate, '. Skipping')
            mel_spec = mel_spectrogram(waveform)[0]
            self.features.append(torch.transpose(mel_spec,0,1))

        self.label_map =  dict((v,k) for k,v in tmp_label_map.items())


        
    def __len__(self):
        return len(self.features)
    
    
    def __getitem__(self, idx):
        features = self.features[idx]                
        label = self.labels[idx]
        return features, label

    def get_sample_shape(self):
        return self.features[0].shape

    def get_num_classes(self):
        return len(self.label_map)

    
    def plot_sample(self,specgram, title=None, ylabel="mel_freq", ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
        plt.show()


import sys
def main():
    dataset = AudioDataset(sys.argv[1], audio_path = sys.argv[2])
    
    
if __name__ == '__main__':
    main()
