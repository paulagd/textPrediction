import torch
import torch.nn as nn
from IPython import embed


class CharRNN(nn.Module):

    def __init__(self, dictionary_len, dropout, hidden_size, layers, embedding_len=32, device='cpu', lr=0.001):
        super().__init__()
        self.embedding = nn.Embedding(dictionary_len, embedding_len, padding_idx=-1)
        self.lstm = nn.LSTM(embedding_len, hidden_size, batch_first=True, bidirectional=False,
                            num_layers=layers, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, dictionary_len)
        # self.fc2 = nn.Linear(hidden_size, diccionary_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.hidden_size = hidden_size
        self.device = device
        self.lr = lr

    def forward(self, x, prev_state=None):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        embeding = self.embedding(x)
        # embeding = self.dropout(embeding)

        # get the outputs and the new hidden state from the lstm
        # B x T x embeding_len
        if prev_state:
            output, hidden_state = self.lstm(embeding, prev_state)
        else:
            output, hidden_state = self.lstm(embeding)

        output = self.dropout(output)
        output = self.fc1(output)

        # return the final output and the hidden state
        return output, hidden_state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))
