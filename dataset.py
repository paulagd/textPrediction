# -*- coding: UTF-8 -*-
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from utils import ALPHABET, unicode_to_ascii, letter_to_index
from IPython import embed
from nltk.tokenize import sent_tokenize, word_tokenize


class TextDataset(Dataset):
    def __init__(self, data, labels, max_len=35, transform=None):

        self.data = data
        self.labels = labels
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        # return len(self.labels) - self.max_len-1
        return len(self.labels) - self.max_len

    def __getitem__(self, idx):

        # embed()
        # char_seq = unicode_to_ascii(self.data[idx:idx+self.max_len], ALPHABET)
        char_seq = self.data[idx:idx+self.max_len]
        # char_label_seq = unicode_to_aself.data[idx:idx+self.max_len], ALPHABET)
        # char_label_seq = self.data[(idx+1):(idx+1)+self.max_len]
        char_label_seq = self.labels[idx:idx + self.max_len]

        encoded = [letter_to_index(char, ALPHABET) for char in char_seq]
        encoded_label = [letter_to_index(char, ALPHABET) for char in char_label_seq]

        # embed()
        encoded_tensor = torch.tensor(encoded, dtype=torch.long)
        encoded_labels = torch.tensor(encoded_label, dtype=torch.long)
        # encoded = np.array([letter_to_index(char, ALPHABET) for char in text])
        # embed()
        return encoded_tensor, encoded_labels

        
