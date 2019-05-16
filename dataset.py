# -*- coding: UTF-8 -*-
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from utils import ALPHABET, unicode_to_ascii, letter_to_index
from IPython import embed
from nltk.tokenize import sent_tokenize, word_tokenize


class TextDataset(Dataset):
    def __init__(self, char, data, labels, max_len=35, transform=None):

        self.data = data
        self.labels = labels
        self.max_len = max_len
        self.transform = transform
        self.charSplit = char

    def __len__(self):
        # return len(self.labels) - self.max_len-1
        return len(self.labels) - self.max_len

    def __getitem__(self, idx):

        if self.charSplit:
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
            return encoded_tensor, encoded_labels

        else:
            text = self.data.replace("\n", " ")
            text = text.split()
            # Each tuple is ([ word_i-2, word_i-1 ], target word)
            # trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            #             for i in range(len(test_sentence) - 2)]
            # print the first 3, just so you can see what they look like
            # print(trigrams[:3])
            vocab = set(text)
            word_to_ix = {word: i for i, word in enumerate(vocab)}
            # sentences = []
            # for i in sent_tokenize(text):
            #     temp = []
            #
            #     # tokenize the sentence into words
            #     for j in word_tokenize(i):
            #         temp.append(j.lower())
            #
            #     sentences.append(temp)




