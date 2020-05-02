import os
import torch
import pandas
import string
from Generators.DataGenerator import DataGenerator
from NeuralNetwork.DAE import DenoisingAutoEncoder
from NeuralNetwork.AuxClassifier import AuxClassifier
from NeuralNetwork.CharacterClassifier import CharacterClassifier
from Constant import *


class Pipeline(torch.nn.Module):
    def __init__(self, name: str, hidden_sz: int = 256, num_layers: int = 6):
        super(Pipeline, self).__init__()
        self.session_name = name
        printable_lst = [c for c in string.printable] + [EOS, PAD]
        self.data_generator = DataGenerator()
        self.character_classifier = CharacterClassifier(
            printable_lst, CHARACTER_CLASSIFICATIONS)
        self.first_DAE = DenoisingAutoEncoder(
            INPUT, DAE_OUTPUT, hidden_sz, num_layers)
        self.last_DAE = DenoisingAutoEncoder(
            INPUT, DAE_OUTPUT, hidden_sz, num_layers)
        self.title_classifier = AuxClassifier(
            printable_lst, hidden_sz, len(TITLES), num_layers)
        self.suffix_classifier = AuxClassifier(
            printable_lst, hidden_sz, len(SUFFIXES), num_layers)
        self.to(DEVICE)

    def train(self, batch_sz: int, iterations: int):
        for i in range(iterations):
            full, character_classifications = self.data_generator.generateFullName()
            first, noised_first = self.data_generator.generateName(self.data_generator.fn_generator)
            last, noised_last = self.data_generator.generateName(self.data_generator.ln_generator)
            title, noised_title = self.data_generator.generateAux(TITLES)
            suffix, noised_suffx = self.data_generator.generateAux(SUFFIXES)

    def train_character_classifier(self, src: list, trg: list):
        max_len = max(src, key=len)
        return False

    def train_aux_classifiier(self, src: list, trg: list):
        max_src_len = max(src, key=len)
        max_trg_len = max(trg, key=len)
        return False

    def train_DAE(self, src: list, trg: list):
        max_src_len = max(src, key=len)
        max_trg_len = max(trg, key=len)
        return False

    def test(self, df: pandas.DataFrame):
        return False

    def test_name(self, name):
        return False

    def save_checkpoint(self, folder: str = 'Weights'):
        fp = os.path.join(folder, self.name)

        if not os.path.exists(folder):
            os.mkdir(folder)

        content = {'weights': self.state_dict()}

        torch.save(content, fp)
