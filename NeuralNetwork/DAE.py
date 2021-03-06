import torch
import torch.nn as nn
from Constant import *


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, input: list, output: list, hidden_sz: int, num_layers: int, embed_dim: int = 8, drop_out: float = 0.1):
        super().__init__()
        self.input = input
        self.output = output
        self.input_sz = len(input)
        self.output_sz = len(output)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.hidden_sz = hidden_sz

        self.encoder_embed = nn.Embedding(self.input_sz, embed_dim)
        self.encoder_lstm = nn.LSTM(
            embed_dim, hidden_sz, num_layers)
        self.decoder_embed = nn.Embedding(self.output_sz, embed_dim)
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_sz, num_layers)
        self.decoder_fc1 = nn.Linear(hidden_sz, self.output_sz)
        self.sigmoid = nn.Sigmoid()
        self.decoder_fc2 = nn.Linear(self.output_sz, self.output_sz)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(drop_out)

        self.to(DEVICE)

    def encode(self, names: torch.Tensor, lengths: torch.Tensor):
        batch_sz = lengths.shape[0]
        embedded_input = self.encoder_embed(names)
        pps_input = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_input, lengths, enforce_sorted=False)
        output, hidden = self.encoder_lstm(
            pps_input, self.init_hidden(batch_sz))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        return output, hidden

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        embedded_input = self.decoder_embed(input)
        output, hidden = self.decoder_lstm.forward(embedded_input, hidden)
        fc1_output = self.decoder_fc1(output)
        sig_output = self.sigmoid(fc1_output)
        fc2_output = self.decoder_fc2(sig_output)
        do_output = self.dropout(fc2_output)
        probs = self.softmax(do_output)

        return probs, hidden

    def init_hidden(self, batch_sz):
        return (torch.zeros(self.num_layers, batch_sz, self.hidden_sz).to(DEVICE),
                torch.zeros(self.num_layers, batch_sz, self.hidden_sz).to(DEVICE))
