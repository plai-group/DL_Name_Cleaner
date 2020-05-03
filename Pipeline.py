import os
import torch
import pandas
import string
from Generators.DataGenerator import DataGenerator
from NeuralNetwork.DAE import DenoisingAutoEncoder
from NeuralNetwork.AuxClassifier import AuxClassifier
from NeuralNetwork.CharacterClassifier import CharacterClassifier
from Utilities.Utilities import convertToIdxList
from Constant import *


class Pipeline(torch.nn.Module):
    def __init__(self, name: str, hidden_sz: int = 256, num_layers: int = 6, learning_rate: float = 0.00005):
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
            printable_lst, hidden_sz, TITLES, num_layers)
        self.suffix_classifier = AuxClassifier(
            printable_lst, hidden_sz, SUFFIXES, num_layers)

        self.learning_rate = learning_rate
        self.to(DEVICE)

    def train(self, batch_sz: int, iterations: int):
        for i in range(iterations):
            fullnames, char_classes = [], []
            firsts, noised_firsts = [], []
            lasts, noised_lasts = [], []
            titles, noised_titles = [], []
            suffixes, noised_suffixes = [], []

            for j in range(batch_sz):
                full, character_classifications = self.data_generator.generateFullName()
                first, noised_first = self.data_generator.generateName(
                    self.data_generator.fn_generator)
                last, noised_last = self.data_generator.generateName(
                    self.data_generator.ln_generator)
                title, noised_title = self.data_generator.generateAux(TITLES)
                suffix, noised_suffix = self.data_generator.generateAux(
                    SUFFIXES)

                fullnames.append(full)
                char_classes.append(character_classifications)
                firsts.append(first)
                noised_firsts.append(noised_first)
                lasts.append(last)
                noised_lasts.append(noised_last)
                titles.append(title)
                noised_titles.append(noised_title)
                suffixes.append(suffix)
                noised_suffixes.append(noised_suffix)

            self.train_character_classifier(fullnames, char_classes)
            self.train_aux_classifier(
                self.title_classifier, titles, noised_titles)
            self.train_aux_classifier(
                self.suffix_classifier, suffixes, noised_suffixes)
            self.train_DAE(self.first_DAE, firsts, noised_firsts)
            self.train_DAE(self.last_DAE, lasts, noised_lasts)

    def train_character_classifier(self, src: list, trg: list):
        criterion = torch.nn.CrossEntropyLoss
        optimizer = torch.optim.Adam(
            self.character_classifier.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()

        max_len = max(src, key=len)
        in_vocab = self.character_classifier.input
        out_vocab = self.character_classifier.output
        input = torch.LongTensor(convertToIdxList(
            src, in_vocab, max_len)).to(DEVICE)
        length = torch.LongTensor([max_len] * len(src)).to(DEVICE)
        trg = torch.LongTensor(convertToIdxList(
            trg, out_vocab, max_len)).to(DEVICE)

        outputs, hidden = self.character_classifier.encode(input, length)

        optimizer.step()
        return False

    def train_aux_classifier(self, classifier: AuxClassifier, src: list, trg: list):
        criterion = torch.nn.CrossEntropyLoss
        optimizer = torch.optim.Adam(
            classifier.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()

        max_src_len = max(src, key=len)
        in_vocab = classifier.input
        out_vocab = classifier.output

        input = torch.LongTensor(convertToIdxList(
            src, in_vocab, max_src_len)).to(DEVICE)
        trg = torch.LongTensor(convertToIdxList(trg, out_vocab, 1)).to(DEVICE)

        optimizer.step()

        return False

    def train_DAE(self, dae: DenoisingAutoEncoder, src: list, trg: list):
        criterion = torch.nn.CrossEntropyLoss
        optimizer = torch.optim.Adam(dae.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()

        max_src_len = max(src, key=len)
        max_trg_len = max(trg, key=len)
        in_vocab = dae.input
        out_vocab = dae.output

        input = torch.LongTensor(convertToIdxList(
            src, in_vocab, max_src_len)).to(DEVICE)
        trg = torch.LongTensor(convertToIdxList(
            trg, out_vocab, max_trg_len, True)).to(DEVICE)

        optimizer.step()

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
