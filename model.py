import torch
import torch.nn as nn
from IPython import embed

class CharRNN(nn.Module):

    def __init__(self, dictionary_len, dropout, hidden_size, layers, embedding_len=64, device='cpu', lr=0.001):
        super().__init__()
        self.embedding = nn.Embedding(dictionary_len, embedding_len, padding_idx=0)
        self.lstm = nn.LSTM(embedding_len, hidden_size, batch_first=True, bidirectional=False,
                            num_layers=layers, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, dictionary_len)
        # self.fc2 = nn.Linear(hidden_size, diccionary_len)
        self.layers = layers
        self.hidden_size = hidden_size
        self.device = device
        self.lr = lr

    def forward(self, x, prev_state):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        embeding = self.embedding(x)
        # get the outputs and the new hidden state from the lstm
        # B X T X embeding_len
        output, hidden_state = self.lstm(embeding, prev_state)
        # _, (h, _) = self.lstm(word)
        # Stack up LSTM outputs using view
        # output = output.contiguous().view(-1, self.hidden_size)

        # h = h.view(self.layers, 2, -1, self.hidden_size)
        # h_last = h[-1]

        # put x through the fully-connected layer
        output = self.fc1(output)
        # output = self.fc2(output)

        # return the final output and the hidden state
        return output, hidden_state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

    # def init_hidden(self, batch_size):
    #     ''' Initializes hidden state '''
    #     # Create two new tensors with sizes n_layers x batch_size x n_hidden,
    #     # initialized to zero, for hidden state and cell state of LSTM
    #     weight = next(self.parameters()).data
    #
    #     hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device),
    #               weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(self.device))
    #
    #     return hidden