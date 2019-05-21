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


def sample(net, size, device='cuda', prime='The', top_k=None):

    net.to(device)

    net.eval()  # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


# Defining a method to generate the next character
def predict(net, char, device='cuda', h=None, top_k=None):
    ''' Given a character, predict the next character.
        Returns the predicted character and the hidden state.
    '''

    # tensor inputs
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)
    inputs = inputs.to(device)

    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)

    # get the character probabilities
    p = F.softmax(out, dim=1).data
    p = p.to(device)

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    # return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h


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
    checkpoint = torch.load("weights/190517163324/checkpoint_16.pt",
                            map_location=('cpu' if device != 'cuda' else None))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    #
    # predict()
    # for _ in range(100):
    #         ix = torch.tensor([[choice]]).to(device)
    #         output, (state_h, state_c) = net(ix, (state_h, state_c))
    #
    #         _, top_ix = torch.topk(output[0], k=top_k)
    #         choices = top_ix.tolist()
    #         choice = np.random.choice(choices[0])
    #         words.append(int_to_vocab[choice])
    #
    #     print(' '.join(words))
    words = []

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
        # ???????
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
