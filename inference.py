import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TextDataset
import argparse
from datetime import datetime
from model import CharRNN
from utils import get_labels_text_prediction, index_to_letter, ALPHABET, unicode_to_ascii, letter_to_index
import numpy as np
import os
from tqdm import trange
from IPython import embed
import heapq


def sample(model, device, index_sample, range_seq=100, idx_prob=0):

    letter = torch.LongTensor([index_sample]).reshape(1, 1)
    letter = letter.to(device)

    pred, state = model(letter)

    words = []
    embed()
    if idx_prob == 0:
        pred = torch.argmax(pred, dim=-1)
    else:
        pred = heapq.nlargest(10, range(len(pred)), key=pred.__getitem__)[idx_prob]

    words.append(index_to_letter(pred.squeeze().item(), ALPHABET))

    for _ in range(range_seq):
        pred, state = model(pred, state)
        pred = torch.argmax(pred, dim=-1)
        words.append(index_to_letter(pred.squeeze().item(), ALPHABET))

    return ''.join(words)


def inference():

    # Declaring the hyperparameters
    hidden_size = 128
    # seq_length = 100
    # opt = parse_args()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)

    # y_test = get_labels_text_prediction(x_test)
    # test_dataset = TextDataset(x_test, y_test)

    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     pin_memory=device == 'cuda',
    #     batch_size=batch_size,
    #     shuffle=False)

    model_params = {'dictionary_len': len(ALPHABET),
                    'dropout': 0,
                    'hidden_size': hidden_size,
                    'layers': 1,
                    'embedding_len': 32,
                    'device': device,
                    'lr': 0.001
                    }

    model = CharRNN(**model_params).to(device)
    checkpoint = torch.load("weights/190517163324/checkpoint_16.pt",
                            map_location=('cpu' if device != 'cuda' else None))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    # idx_prob GETS 0 for max prob / 1 for second max prob / etc...
    predicted_words = sample(model, device, 12, 500, idx_prob=1)
    print(' '.join(predicted_words))

    #
    # for i, (x, y) in enumerate(test_loader):
    #
    #     if i == 1:
    #         break
    #     if i == len(test_loader) - 1:
    #         # print("FER PADDING -  DE MOMENT NO VA")
    #         continue
    #
    #     model.eval()
    #     state_h, state_c = model.zero_state(1)
    #     # state_h, state_c = model.zero_state(opt.batch_size)
    #     state_h = state_h.to(device)
    #     state_c = state_c.to(device)
    #
    #     x = x.to(device)
    #     # ???????
    #     output, (state_h, state_c) = model(x[0].unsqueeze(0), (state_h, state_c))
    #
    #     _, top_x = torch.topk(output[0], k=top_k)
    #     choices = top_x.tolist()
    #     choice = np.random.choice(choices)
    #     pred_word = index_to_letter(choice, ALPHABET)
    #     words.append(pred_word)
    #
    #     for _ in range(100):
    #
    #         char = unicode_to_ascii(pred_word, ALPHABET)
    #         encoded = letter_to_index(char, ALPHABET)
    #         new_x = torch.tensor([encoded], dtype=torch.long).to(device)
    #         output, (state_h, state_c) = model(new_x, (state_h, state_c))
    #
    #         _, top_x = torch.topk(output[0], k=top_k)
    #         choices = top_x.tolist()
    #         choice = np.random.choice(choices)
    #         pred_word = index_to_letter(choice, ALPHABET)
    #         words.append(pred_word)


if __name__ == '__main__':
    inference()
