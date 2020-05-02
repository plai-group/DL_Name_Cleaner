import torch
from Constant import *


class CharacterClassifier(torch.nn.Module):
    def __init__(self, input: list, output: list, hidden_sz: int = 256, num_layers: int = 4, embed_dim: int = 4, dropout: float = 0.1):
        super(CharacterClassifier, self).__init__()
        self.input = input
        self.output = output
        self.input_sz = len(input)
        self.output_sz = len(output)
        self.embed = torch.nn.Embedding(self.input_sz, embed_dim)
        self.lstm = torch.nn.LSTM(
            self.input_sz, hidden_sz, num_layers, bidirectional=True)
        self.fc1 = torch.nn.Linear(hidden_sz * 4, hidden_sz)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(hidden_sz, self.output_sz)
        self.dropout = torch.nn.Dropout(dropout)

    def encode(self, input: torch.Tensor, lengths: torch.Tensor):
        hidden = self.init_hidden()
        embedded_input = self.embed(input)
        batch_sz = embedded_input.shape[1]

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_input, lengths)

        output, hidden = self.lstm.forward(packed_input, hidden)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)

        return output, hidden

    def init_hidden(self, batch_sz):
        return (torch.zeros(self.num_layers * 2, batch_sz, self.hidden_sz).to(DEVICE),
                torch.zeros(self.num_layers * 2, batch_sz, self.hidden_sz).to(DEVICE))
