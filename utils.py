import unicodedata
import string
import numpy as np
import torch
from IPython import embed

# ALPHABET = string.ascii_letters + string.digits + string.whitespace + ":;,.¡!?¿()"

ALPHABET = "yIYKz.\'l:;)OrCiJGPQ\n*nWEXkc[(ojmfp-dxwUSeBVsu,qMTRA]gL b!vNDFa\"h?Ht"
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


def do_inference_test(first_sentence, model, device, range_seq=100):
    words = []
    top_k = 50
    first_sentence = first_sentence.transpose(2, 1)
    state_h, state_c = model.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)

    first_sentence = first_sentence.to(device)
    
    _, top_x = torch.topk(first_sentence, k=top_k)
    choices = top_x.tolist()[0]
    r = np.random.randint(len(choices))
    choice = choices[r]
    pred_word = [index_to_letter(c, ALPHABET) for c in choice]
    pred_word = ''.join(pred_word)
    words.append(pred_word)

    for _ in range(range_seq):
        # char = unicode_to_ascii(pred_word, ALPHABET)
        encoded = [letter_to_index(w, ALPHABET) for w in pred_word]
        new_x = torch.tensor([encoded], dtype=torch.long).to(device)
        output, (state_h, state_c) = model(new_x, (state_h, state_c))
        # output = output.transpose(1, 2)

        _, top_x = torch.topk(output[0], k=top_k)
        choices = top_x.tolist()
        r = np.random.randint(len(choices))
        choice = choices[r]
        pred_word = [index_to_letter(c, ALPHABET) for c in choice]
        pred_word = ''.join(pred_word)
        words.append(pred_word)

    return ' '.join(words)


# NO NUESTRO
def unicode_to_ascii(s, all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def letter_to_index(letter, all_letters):
    return all_letters.find(letter)


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
