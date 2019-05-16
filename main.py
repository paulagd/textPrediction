import argparse
from utils import split_data, set_alphabet, ALPHABET
from train import train
from inference import inference
from IPython import embed


def parse_args():
    # srun -p veu -c8 --mem 30G python main.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="enables cuda")
    parser.add_argument("--onlytrain", action="store_true", default=False, help="enables only training")
    # parser.add_argument("--num_workers", type=int, default=4, help="Num cpu workers")
    # parser.add_argument("--seed", type=int, default=5, help="Random seed")
    parser.add_argument("--hidden_size", type=int, default=512, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=128, help="Random seed")
    parser.add_argument("--layers", type=int, default=1, help="Random seed")
    parser.add_argument("--dropout", type=float, default=0, help="dropout parameter")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate parameter")
    parser.add_argument("--nologs", action="store_true", help="enables no tensorboard")
    parser.add_argument("--inference", action="store_true", help="enables INFERENCE MODE")
    parser.add_argument("--char", action="store_true", help="enables char embedding MODE")
    parser.add_argument("--scheduler", action="store_true", help="enables char embedding MODE")
    # parser.add_argument(
    #     "--outc", default="./checkpoints/", help="folder to output model checkpoints"
    # )

    options = parser.parse_args()
    print(options)
    return options


if __name__ == '__main__':

    opt = parse_args()
    # IDEA: Split just in TRAIN and TEST because the final test will be just
    # by writing a letter, predict the whole sentence --> we dont need a test set

    # x_train, x_test, x_val = split_data('./11.txt', 0.8, opt.onlytrain)
    x_train, x_test, x_val = split_data('./11.txt', 0.8, False)
    # embed()
    # dictionary_len = set_alphabet(x_train)
    dictionary_len = len(ALPHABET)

    if opt.inference:
        inference(opt, x_test, dictionary_len)
    else:
        train(opt, x_train, x_val, dictionary_len)
        # train(opt, x_train, x_test, dictionary_len)












