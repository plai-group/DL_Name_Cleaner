import os
from Constant import *
from NeuralNetwork.LSTM import LSTM
from Utilities.Config import *


class NameGenerator():
    def __init__(self, config_path: str, weights_path: str):
        super().__init__()
        config = load_json(config_path)
        self.hidden_sz = config['hidden_size']
        self.num_layers = config['num_layers']
        self.input = config['input']
        self.output = config['output']
        self.embed_sz = config['embed_dim']
        self.input_sz = len(self.input)
        self.output_sz = len(self.output)
        self.SOS = config['SOS']
        self.PAD = config['PAD']
        self.EOS = config['EOS']
        self.probs = config['probs']

        self.lstm = LSTM(
            self.input_sz, self.hidden_sz, self.output_sz, self.embed_sz, self.num_layers)

        if weights_path is not None:
            self.load_weights(weights_path)

    def load_weights(self, path):
        if not os.path.exists(path):
            raise Exception(f"Path does not exist: {path}")
        self.lstm.load_state_dict(torch.load(
            path, map_location=DEVICE)['weights'])
        self.lstm.eval()

    def forward(self, input: torch.Tensor, length: torch.Tensor, hidden_state: torch.Tensor = None):
        with torch.no_grad():
            if hidden_state is None:
                hidden_state = self.lstm.initHidden(1)

            output, hidden = self.lstm.forward(input, length, hidden_state)
            return output, hidden
    
    def generateName(self, length: int):
        input = torch.LongTensor([self.input.index(self.SOS)]).unsqueeze(0).to(DEVICE)
        length_input = torch.LongTensor([length]).unsqueeze(0).to(DEVICE)
        hidden = None

        ret = ''
        for i in range(length):
            output, hidden = self.forward(input, length_input, hidden)
            sample = int(torch.distributions.Categorical(
                output.exp()).sample().item())
            sample_char = self.output[sample]
            input = torch.LongTensor([self.input.index(sample_char)]).unsqueeze(0).to(DEVICE)

            ret += sample_char

        return ret
    
    def sampleName(self):
        probs_tensor = torch.zeros(max(int(k) for k, v in self.probs.items()) + 1)

        for key, value in self.probs.items():
            probs_tensor[int(key)] = value

        length = int(torch.distributions.Categorical(probs_tensor).sample().item())

        return self.generateName(length)