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


def inference(opt, x_test, dictionary_len):

    # Declaring the hyperparameters
    batch_size = opt.batch_size
    seq_length = 100
    epochs = 100 # start smaller if you are just testing initial behavior
    top_k = 50
    # opt = parse_args()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)

    y_test = get_labels_text_prediction(x_test)
    test_dataset = TextDataset(x_test, y_test)

    test_loader = DataLoader(
        dataset=test_dataset,
        pin_memory=device == 'cuda',
        batch_size=batch_size,
        shuffle=False)

    model_params = {'dictionary_len': dictionary_len,
                    'dropout': opt.dropout,
                    'hidden_size': opt.hidden_size,
                    'layers': opt.layers,
                    'embedding_len': 64,
                    'device': device,
                    'lr': opt.lr
                    }

    model = CharRNN(**model_params).to(device)
    checkpoint = torch.load("weights/190509160128/checkpoint_16.pt",
                            map_location=('cpu' if device != 'cuda' else None))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    words = []
    criterion = nn.CrossEntropyLoss()

    for i, (x, y) in enumerate(test_loader):

        if i == 1:
            break
        if i == len(test_loader) - 1:
            # print("FER PADDING -  DE MOMENT NO VA")
            continue

        model.eval()
        state_h, state_c = model.zero_state(1)
        # state_h, state_c = model.zero_state(opt.batch_size)
        state_h = state_h.to(device)
        state_c = state_c.to(device)

        x = x.to(device)
        output, (state_h, state_c) = model(x[0].unsqueeze(0), (state_h, state_c))

        _, top_x = torch.topk(output[0], k=top_k)
        choices = top_x.tolist()
        choice = np.random.choice(choices)
        pred_word = index_to_letter(choice, ALPHABET)
        words.append(pred_word)

        for _ in range(100):

            char = unicode_to_ascii(pred_word, ALPHABET)
            encoded = letter_to_index(char, ALPHABET)
            new_x = torch.tensor([encoded], dtype=torch.long).to(device)
            output, (state_h, state_c) = model(new_x, (state_h, state_c))

            _, top_x = torch.topk(output[0], k=top_k)
            choices = top_x.tolist()
            choice = np.random.choice(choices)
            pred_word = index_to_letter(choice, ALPHABET)
            words.append(pred_word)

        print(' '.join(words))







