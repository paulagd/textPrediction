import unicodedata
import string
import numpy as np
import torch
from IPython import embed

# ALPHABET = "#yIYKz.\'l:;)OrCiJGPQ\n*nWEXkc[(ojmfp-dxwUSeBVsu,qMTRA]gL b!vNDFa\"h?Ht"
# ALPHABET = '#abcdefghijklmnñopqrstuvwxyz[()] *-?\'.,;!\n\"'
ALPHABET = '#\n !"$%\'()*,-./0123456789:;?@[]_abcdefghñijklmnopqrstuvwxyz'
# global ALPHABET
# ALPHABET = 'a'


def split_data(dataset_path, train_partition, only_train=False):
    '''
    Takes the dataset path and split it into train and test set.
        - If val == True, it splits the 30% of the training test into validation as well.
    '''
    with open(dataset_path, 'r') as f:
        dataset = f.read()  # Alice

    aux = round(len(dataset) * train_partition)
    tr_set = dataset[:aux]
    tst_set = dataset[aux:]
    val_set = []

    assert len(tr_set) + len(tst_set) == len(dataset)

    if not only_train:
        val_aux = int(0.3 * len(tr_set))
        val_set = tr_set[:val_aux]
        tr_set = tr_set[val_aux:]

    return tr_set, tst_set, val_set


def quitar_accentos(s):
    import re
    from unicodedata import normalize
    # -> NFD y eliminar diacríticos
    s = re.sub(
            r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1",
            normalize("NFD", s), 0, re.I
        )
    # -> NFC
    s = normalize('NFC', s)

    return s


def split_data_without_test(dataset_path, idioma="en"):
    '''
    Takes the dataset path and split it into train and test set.
        - If val == True, it splits the 30% of the training test into validation as well.
    '''
    if idioma != 'en':
        with open(dataset_path, 'r', encoding="ISO-8859-1") as f:
            dataset = f.read()  # Alice
        dataset = dataset.lower()
        dataset = quitar_accentos(dataset)
    else:
        with open(dataset_path, 'r') as f:
            dataset = f.read()  # Alice
        dataset = dataset.lower()

    aux = round(len(dataset) * 0.75)
    train_set = dataset[:aux]
    val_set = dataset[aux:]

    assert len(train_set) + len(val_set) == len(dataset)

    return train_set, val_set


def get_labels_text_prediction(x, padding=None):
    y = x[1:]
    if padding:
        y += padding
    else:
        y += x[0]
    return y


def set_alphabet(train_set):
    global ALPHABET
    ALPHABET = ''.join(set(train_set))
    return len(ALPHABET)


def inference_prediction(model, device, range_seq=100):

    letter = torch.LongTensor([38]).reshape(1, 1)
    letter = letter.to(device)

    pred, state = model(letter)

    words = []
    pred = torch.argmax(pred, dim=-1)
    words.append(index_to_letter(pred.squeeze().item(), ALPHABET))

    for _ in range(range_seq):
        pred, state = model(pred, state)
        pred = torch.argmax(pred, dim=-1)
        words.append(index_to_letter(pred.squeeze().item(), ALPHABET))

    return ''.join(words)


# NO NUESTRO
def unicode_to_ascii(s, all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def letter_to_index(letter, all_letters):
    value = all_letters.find(letter)
    if value == -1:
        return 0
    else:
        return value


def index_to_letter(idx, all_letters):
    return all_letters[idx]


def get_dictionary(dataset):
    return tuple(set(dataset))


def load_and_prepare_data(dataset):
    # read and prepare the data
    with open(dataset, 'r') as f: text = f.read()  # Alice
    # with open('./1399-0.txt', 'r') as f: text = f.read() # Ana Karenina
    all_letters = string.ascii_letters + " .,;'"
    text = unicode_to_ascii(text, all_letters)
    # encode the text, using the character to integer function
    encoded = np.array([letter_to_index(char, all_letters) for char in text])
    n_letters = len(all_letters)

    return text, encoded, n_letters


# Defining method to encode one hot labels
def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


# def do_inference_test(first_sentence, model, device, range_seq=10):
#     words = []
#     top_k = 10
#
#
#     # first_sentence = first_sentence.transpose(2, 1)
#     # state_h, state_c = model.zero_state(1)
#     # state_h = state_h.to(device)
#     # state_c = state_c.to(device)
#
#     first_sentence = first_sentence.to(device)
#     # [1,40,1]
#     max_y = torch.max(first_sentence, dim=2)[1].squeeze()
#     # _, top_x = torch.topk(first_sentence, k=top_k)
#     # choices = top_x.tolist()[0]
#     # r = np.random.randint(len(choices))
#     # choice = choices[r]
#
#     pred_word = [index_to_letter(c, ALPHABET) for c in max_y]
#     pred_word = ''.join(pred_word)
#     words.append(pred_word)
#
#     for _ in range(range_seq):
#         # char = unicode_to_ascii(pred_word, ALPHABET)
#         encoded = [letter_to_index(w, ALPHABET) for w in pred_word]
#         new_x = torch.tensor([encoded], dtype=torch.long).to(device)
#         output, (state_h, state_c) = model(new_x)
#         # output = output.transpose(1, 2)
#
#         max_y = torch.max(output, dim=2)[1].squeeze()
#
#         pred_word = [index_to_letter(c, ALPHABET) for c in max_y]
#         pred_word = ''.join(pred_word)
#         words.append(pred_word)
#
#     return ' '.join(words)
