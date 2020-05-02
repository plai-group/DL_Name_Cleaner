import os
import torch
import pandas
import string
from Generators.FullNameGenerator import FullNameGenerator
from NeuralNetwork.DAE import DenoisingAutoEncoder
from NeuralNetwork.AuxClassifier import AuxClassifier
from NeuralNetwork.CharacterClassifier import CharacterClassifier
from Utilities.Noiser import noise_name
from Constant import *

NOISE_CHARS = [c for c in string.ascii_letters] + \
    [c for c in string.digits] + ['_', '\'', ':']


class Pipeline(torch.nn.Module):
    def __init__(self, name: str, hidden_sz: int = 256, num_layers: int = 6):
        super(Pipeline, self).__init__()
        self.session_name = name
        printable_lst = [c for c in string.printable] + [EOS, PAD]
        self.data_generator = FullNameGenerator()
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
            # tuple idx=0 of data is target and idx=1 is the src
            data = self.generate_batch_data(batch_sz)

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

    def generate_batch_data(self, batch_sz: int, noise_percent: float = 0.1):
        firsts_lst, middles_lst, lasts_lst, title_lst, suffix_lst, full_lst, noised_firsts_lst, noised_middles_lst, noised_lasts_lst, noised_titles_lst, noised_suffix_lst, classification_lst = [
        ], [], [], [], [], [], [], [], [], [], [], []

        for i in range(batch_sz):
            format_idx = int(torch.distributions.Categorical(torch.FloatTensor(
                [1/len(NAME_FORMATS)] * len(NAME_FORMATS))).sample().item())

            full_noised_name = NAME_FORMATS[format_idx]
            character_classification = full_noised_name

            first, middles, last, title, suffix = self.generate_components(
                format_idx)

            noised_first = noise_name(
                first, NOISE_CHARS, noise_percent)

            noised_middles = []
            for name in middles:
                noised_middles.append(noise_name(
                    name, NOISE_CHARS, noise_percent))

            noised_last = noise_name(last, NOISE_CHARS, noise_percent)

            noised_title = noise_name(
                title, NOISE_CHARS, noise_percent) if title is not None else None

            noised_suffix = noise_name(
                suffix, NOISE_CHARS, noise_percent) if suffix is not None else None

            full_noised_name = full_noised_name.replace('{f}', noised_first)
            character_classification = character_classification.replace(
                '{f}', 'f' * len(noised_first))

            full_noised_name = full_noised_name.replace('{l}', noised_last)
            character_classification = character_classification.replace(
                '{l}', 'l' * len(noised_last))

            if len(noised_middles) > 0:
                middles_combined = ''
                middles_classification = ''

                for i in range(len(noised_middles)):
                    name = noised_middles[i]

                    if len(name) > 1 and i < len(noised_middles) - 1:
                        middles_combined += name + ' '
                        middles_classification += 'm' * len(name) + ' '
                    elif len(name) < 2:
                        middles_combined += name + '. '
                        middles_classification += 'm' * len(name) + '. '
                    else:
                        middles_combined += name
                        middles_classification += 'm' * len(name)

                full_noised_name = full_noised_name.replace(
                    '{m}', middles_combined)
                character_classification = character_classification.replace(
                    '{m}', middles_classification)

            full_noised_name = noised_title + ' ' + \
                full_noised_name if noised_title is not None else full_noised_name
            character_classification = len(
                noised_title) * 't' + '. ' + character_classification if noised_title is not None else character_classification

            full_noised_name = full_noised_name + ' ' + \
                noised_suffix if noised_suffix is not None else full_noised_name
            character_classification = character_classification + ' ' + \
                len(noised_suffix) * \
                's' if noised_suffix is not None else character_classification

            firsts_lst.append(first)
            middles_lst.append(middles)
            lasts_lst.append(last)
            title_lst.append(title)
            suffix_lst.append(suffix)
            noised_firsts_lst.append(noised_first)
            noised_middles_lst.append(noised_middles)
            noised_lasts_lst.append(noised_last)
            full_lst.append(full_noised_name)
            classification_lst.append(character_classification)

        return ({'first': firsts_lst, 'middle': middles_lst, 'last': lasts_lst, 'title': title_lst, 'suffix': suffix_lst, 'class': classification_lst},
                {'first': noised_firsts_lst, 'middle': noised_middles_lst, 'last': noised_lasts_lst, 'title': noised_titles_lst, 'suffix': noised_suffix_lst, 'full': full_noised_name})

    def generate_components(self, format_idx: int):
        has_middle = self.has_middle(format_idx)
        has_title = bool(torch.distributions.Bernoulli(
            torch.FloatTensor([0.25])).sample().item())
        has_suffix = bool(torch.distributions.Bernoulli(
            torch.FloatTensor([0.25])).sample().item())

        first, middles, last, title, suffix = self.data_generator.sampleAllNameComponents(
            has_middle, has_title, has_suffix)

        return first, middles, last, title, suffix

    def has_middle(self, format_idx):
        return '{m}' in NAME_FORMATS[format_idx]

    def save_checkpoint(self, folder: str = 'Weights'):
        fp = os.path.join(folder, self.name)

        if not os.path.exists(folder):
            os.mkdir(folder)

        content = {'weights': self.state_dict()}

        torch.save(content, fp)
