import torch
import torch.nn as nn
import torch.optim as optim

from Constant import *


class AuxClassifier(nn.Module):
    def __init__(self, input: list, hidden_sz: int, output: list, num_layers: int = 4, embed_sz: int = 16,
                 drop_out: float = 0.1):
        super(AuxClassifier, self).__init__()
        self.input = input
        self.input_sz = len(input)
        self.hidden_sz = hidden_sz
        self.output = output
        self.output_sz = len(output)
        self.num_layers = num_layers
        self.embed_sz = embed_sz
        self.fc1_input_sz = (self.hidden_sz * 2 * self.num_layers) * 2

        self.embed = nn.Embedding(
            self.input_sz, self.embed_sz, self.input.index(PAD))
        self.lstm = nn.LSTM(self.embed_sz, self.hidden_sz,
                            num_layers=num_layers, bidirectional=True).to(DEVICE)
        self.fc1 = nn.Linear(self.fc1_input_sz, self.output_sz)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(self.output_sz, self.output_sz)
        self.dropout = nn.Dropout(drop_out)
        self.softmax = nn.Softmax(dim=1)

        self.to(DEVICE)

    def forward(self, input: torch.Tensor, lengths: torch.IntTensor):
        batch_sz = lengths.shape[0]
        embedded_input = self.embed(input)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_input, lengths, enforce_sorted=False)

        output, (hidden, cell) = self.lstm.forward(
            packed_input, self.init_hidden(batch_sz))

        # hidden is a tuple of 2 hidden tensors that are a forward and backward tensor in one
        hidden_states = torch.cat((hidden, cell), 2)
        hidden_states = hidden_states.transpose(0, 1).flatten(1, 2)

        output = self.fc1(hidden_states)
        output = self.sigmoid(output)
        output = self.fc2(output)
        output = self.dropout(output)
        probs = self.softmax(output)

        return probs

    def init_hidden(self, batch_sz: int):
        return (torch.zeros(2 * self.num_layers, batch_sz, self.hidden_sz).to(DEVICE),
                torch.zeros(2 * self.num_layers, batch_sz, self.hidden_sz).to(DEVICE))
