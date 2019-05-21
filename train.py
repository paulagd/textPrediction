import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_labels_text_prediction, do_inference_test, pred_dani
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import TextDataset
from datetime import datetime
from model import CharRNN
from tqdm import trange
import os
import random
import numpy as np
from IPython import embed


def train(opt, x_train, x_val, dictionary_len):
    ''' Training a network

        Arguments
        ---------

        net: CharRNN network
        data: training data to train the network (text)
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

    '''
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # x_val = x_train[:1000]
    # Declaring the hyperparameters
    batch_size = opt.batch_size
    seq_length = int(opt.seq_length)
    epochs = 156  # start smaller if you are just testing initial behavior

    # opt = parse_args()
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed_all(0)
    else:
        device = "cpu"
    print(device)

    date = datetime.now().strftime('%y%m%d%H%M%S')
    if opt.nologs:
        writer = SummaryWriter(log_dir=f'logs/nologs/')
    else:
        writer = SummaryWriter(log_dir=f'logs/logs_{date}/')

    y_train = get_labels_text_prediction(x_train)

    train_dataset = TextDataset(x_train, y_train, max_len=seq_length)

    if not opt.onlytrain:
        y_val = get_labels_text_prediction(x_val)
        val_dataset = TextDataset(x_val, y_val, max_len=seq_length)
        val_loader = DataLoader(
            dataset=val_dataset,
            pin_memory=device == 'cuda',
            batch_size=batch_size,
            shuffle=False)

    train_loader = DataLoader(
        dataset=train_dataset,
        pin_memory=device == 'cuda',
        batch_size=batch_size,
        shuffle=True)

    model_params = {'dictionary_len': dictionary_len,
                    'dropout': opt.dropout,
                    'hidden_size': opt.hidden_size,
                    'layers': opt.layers,
                    'embedding_len': 32,
                    'device': device,
                    'lr': opt.lr
                    }

    model = CharRNN(**model_params).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    if opt.scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', cooldown=3, factor=0.5, patience=10)

    global_step = 0
    for j in trange(epochs, desc='T raining LSTM...'):

        for i, (x, y) in enumerate(train_loader):

            if i == len(train_loader) - 1:
                print("FER PADDING -  DE MOMENT NO VA")
                continue
            model.train()

            # embed()
            x = x.to(device)
            y = y.to(device)

            # state_h, state_c = model.zero_state(opt.batch_size)
            # # Transfer data to GPU
            # state_h = state_h.to(device)
            # state_c = state_c.to(device)

            # DELETE PAST GRADIENTS
            optimizer.zero_grad()
            # FORWARD PASS  --> ultim state , (tots)  [ state_h[-1] == pred ]
            pred, (state_h, state_c) = model(x)
            # pred, (state_h, state_c) = model(x, (state_h, state_c))
            # CALCULATE LOSS
            # embed()
            # pred = pred.transpose(1, 2)
            pred2 = pred.view(-1, dictionary_len)
            y2 = y.view(-1)
            loss = criterion(pred2, y2)
            # loss = criterion(pred, y.view(batch_size * seq_length).long())
            loss_value = loss.item()

            # BACKWARD PASS
            loss.backward()
            # MINIMIZE LOSS
            optimizer.step()
            global_step += 1
            if i % 100 == 0:
                writer.add_scalar('train/loss', loss_value, global_step)
                print('[Training epoch {}: {}/{}] Loss: {}'.format(j, i, len(train_loader), loss_value))

        if not opt.onlytrain:
            val_loss = []

            for i, (x, y) in enumerate(val_loader):

                if i == len(val_loader) - 1:
                    # print("FER PADDING -  DE MOMENT NO VA")
                    continue

                model.eval()

                x = x.to(device)
                y = y.to(device)

                # state_h, state_c = model.zero_state(opt.batch_size)
                # state_h = state_h.to(device)
                # state_c = state_c.to(device)

                # NO BACKPROPAGATION
                # FORWARD PASS
                # pred, (state_h, state_c) = model(x, (state_h, state_c))
                pred, (state_h, state_c) = model(x)
                # CALCULATE LOSS
                # pred = pred.transpose(1, 2)

                # pred = [batch x 40 x diccionary_len]
                # y =  [batch x 40]
                pred2 = pred.view(-1, dictionary_len)
                y2 = y.view(-1)
                loss = criterion(pred2, y2)
                # loss = criterion(pred, y)

                val_loss.append(loss.item())

                if i % 50 == 0:
                    print('[Validation epoch {}: {}/{}] Loss: {}'.format(j, i, len(val_loader), loss.item()))

            writer.add_scalar('val/loss', np.mean(val_loss), j)

            if opt.scheduler:
                scheduler.step(np.mean(val_loss))
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], j)

            predicted_words = pred_dani(model, device, 500)
            # output = pred[0].unsqueeze(0)  # [1,diccionary_len, 40]
            # predicted_words = do_inference_test(output, model, device)
            print(predicted_words)
            writer.add_text('val/Generated_Samples', predicted_words, j)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # if j % 5 == 0:
        os.makedirs("weights/{}".format(date), exist_ok=True)
        torch.save(checkpoint, "weights/{}/checkpoint_{}.pt".format(date, j))
