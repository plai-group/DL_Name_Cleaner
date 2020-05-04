import torch
import string
import pandas
from Generators.NameGenerator import NameGenerator
from Constant import DEVICE, TITLES, SUFFIXES, NAME_FORMATS
from Utilities.Noiser import *


class DataGenerator():
    def __init__(self, first_config_pth: str = 'Config/Pretrained/first.json', last_config_pth: str = 'Config/Pretrained/last.json', first_wght_pth: str = 'Weights/Pretrained/first.path.tar', last_wght_pth: str = 'Weights/Pretrained/last.path.tar', noise_prob: float = 0.1):
        super(DataGenerator, self).__init__()
        self.fn_generator = NameGenerator(
            first_config_pth, first_wght_pth)
        self.ln_generator = NameGenerator(
            last_config_pth, last_wght_pth)

        self.fn_df = pandas.read_csv('Data/FirstNames.csv')
        self.ln_df = pandas.read_csv('Data/LastNames.csv')
        self.fn_probs = torch.FloatTensor(
            self.fn_df['probs'].tolist()).to(DEVICE)
        self.ln_probs = torch.FloatTensor(
            self.ln_df['probs'].tolist()).to(DEVICE)

        self.noise_prob = noise_prob
        self.noise_chars = [
            c for c in string.ascii_letters + string.digits] + ['_', '\'', ':']

    def generateFullName(self):
        num_formats = len(NAME_FORMATS)
        probs = torch.FloatTensor([1/num_formats] * num_formats)
        format_idx = int(
            torch.distributions.Categorical(probs).sample().item())
        full_name = NAME_FORMATS[format_idx]
        char_classification = full_name

        first, noised_first = self.generateFirstName()
        last, noised_last = self.generateLastName()

        full_name = full_name.replace('{f}', noised_first)
        char_classification = char_classification.replace(
            '{f}', len(noised_first) * 'f')

        full_name = full_name.replace('{l}', noised_last)
        char_classification = char_classification.replace(
            '{l}', len(noised_last) * 'l')

        if self.hasMiddle(format_idx):
            full_middle = ''
            middle_classification = ''
            middle_count = self.sampleNumMiddleNames()

            for i in range(middle_count):
                is_initial = bool(torch.distributions.Bernoulli(
                    torch.FloatTensor([0.5])).sample().item())

                if is_initial:
                    has_dot = bool(torch.distributions.Bernoulli(
                        torch.FloatTensor([0.5])).sample().item())
                    add_on = '. ' if has_dot else ' '
                    middle, noised_middle = self.generateMiddleInitial()
                    full_middle += noised_middle + add_on
                    middle_classification += (len(noised_middle)
                                              * 'm') + add_on
                else:
                    middle, noised_middle = self.generateFirstName()
                    full_middle += noised_middle + ' '
                    middle_classification += (len(noised_middle) * 'm') + ' '

            full_name = full_name.replace('{m}', full_middle[:-1])
            char_classification = char_classification.replace(
                '{m}', middle_classification[:-1])

        if bool(torch.distributions.Bernoulli(torch.FloatTensor([0.5])).sample().item()):
            title, noised_title = self.generateAux(TITLES)
            full_name = noised_title + '. ' + full_name
            char_classification = (len(noised_title) *
                                   't') + '. ' + char_classification

        if bool(torch.distributions.Bernoulli(torch.FloatTensor([0.5])).sample().item()):
            suffix, noised_suffix = self.generateAux(SUFFIXES)
            full_name += ' ' + noised_suffix
            char_classification += ' ' + (len(noised_suffix) * 's')

        return full_name, char_classification

    def generateFirstName(self):
        return self.generateName(self.fn_generator)

    def generateLastName(self):
        return self.generateName(self.ln_generator)

    def generateName(self, generator: NameGenerator):
        name = generator.sampleName()
        noised_name = noise_name(name, self.noise_chars, self.noise_prob)

        return name, noised_name

    def sampleFirstName(self):
        return self.sampleName(self.fn_df, self.fn_probs)

    def sampleLastName(self):
        return self.sampleName(self.ln_df, self.ln_probs)

    def sampleName(self, df: pandas.DataFrame, probs: torch.Tensor):
        idx = int(torch.distributions.Categorical(probs).sample().item())
        name = df['name'][idx]
        noised = noise_name(name, self.noise_chars, self.noise_prob)

        return name, noised

    def generateAux(self, categories: list):
        length = len(categories)
        probs = torch.FloatTensor([1/length] * length).to(DEVICE)
        sample_idx = int(
            torch.distributions.Categorical(probs).sample().item())
        sample = categories[sample_idx]
        noised_sample = noise_name(sample, self.noise_chars, self.noise_prob)

        return sample, noised_sample

    def sampleNumMiddleNames(self):
        return int(torch.distributions.Categorical(torch.FloatTensor([1/3] * 3)).sample().item()) + 1

    def generateMiddleInitial(self):
        initials = string.ascii_uppercase
        initials_sz = len(initials)
        probs = torch.FloatTensor([1/initials_sz] * initials_sz).to(DEVICE)
        sample = int(torch.distributions.Categorical(probs).sample().item())

        initial = initials[sample]
        noised_initial = noise_initial(initial)

        return initial, noised_initial

    def hasMiddle(self, format_idx):
        template = NAME_FORMATS[format_idx]

        return '{m}' in template
