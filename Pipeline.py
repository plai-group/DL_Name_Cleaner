import os
import torch
import pandas
import string
import matplotlib.pyplot as plt
from Generators.DataGenerator import DataGenerator
from NeuralNetwork.DAE import DenoisingAutoEncoder
from NeuralNetwork.AuxClassifier import AuxClassifier
from NeuralNetwork.CharacterClassifier import CharacterClassifier
from Utilities.Utilities import convertToIdxList
from Constant import *


def doesTensorOnlyHasValues(tensor: torch.Tensor, values: list):
    for i in range(len(tensor)):
        if tensor[i].item() not in values:
            return False

    return True


def parse_name(obs: str, classification: list):
    '''
    Parse name into components based on classification list, which classifies each obs
    index as first, middle, last, title, suffix, sep or pad
    '''
    class_str = ''.join(str(c) for c in classification)
    titles, firsts, middles, lasts, suffixes = [], [], [], [], []

    for i in range(len(CHARACTER_CLASSIFICATIONS)):
        curr_class = CHARACTER_CLASSIFICATIONS[i]
        start = class_str.find(curr_class)
        end = class_str.rfind(curr_class) + 1

        if start < 0:
            continue
        elif curr_class is 't':
            titles = obs[start:end].split()
        elif curr_class is 'f':
            firsts = obs[start:end].split()
        elif curr_class is 'm':
            middles = obs[start:end].split()
        elif curr_class is 'l':
            lasts = obs[start:end].split()
        elif curr_class is 's':
            suffixes = obs[start:end].split()

    return titles, firsts, middles, lasts, suffixes


class Pipeline():
    def __init__(self, name: str, hidden_sz: int = 128, num_layers: int = 3, learning_rate: float = 0.00005):
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

    def train(self, batch_sz: int, iterations: int, save_every: int = 500, plot_every: int = 500) -> int:
        all_losses = []
        total_loss = 0

        for i in range(1, iterations + 1):
            full_noised_names, char_classes = [], []
            firsts, noised_firsts = [], []
            middle_initials, noised_middle_initials = [], []
            lasts, noised_lasts = [], []
            titles, noised_titles = [], []
            suffixes, noised_suffixes = [], []

            for j in range(batch_sz):
                if bool(torch.distributions.Bernoulli(torch.FloatTensor([0.5])).sample().item()):
                    full, character_classifications = self.data_generator.generateFullName()
                else:
                    full, character_classifications = self.data_generator.sampleFullName()

                first, noised_first = self.data_generator.sampleFirstName()
                middle_init, noised_middle_init = self.data_generator.generateMiddleInitial()
                last, noised_last = self.data_generator.sampleLastName()
                title, noised_title = self.data_generator.generateAux(TITLES)
                suffix, noised_suffix = self.data_generator.generateAux(
                    SUFFIXES)

                full_noised_names.append(full)
                char_classes.append(character_classifications)
                firsts.append(first)
                noised_firsts.append(noised_first)
                middle_initials.append(middle_init)
                noised_middle_initials.append(noised_middle_init)
                lasts.append(last)
                noised_lasts.append(noised_last)
                titles.append(title)
                noised_titles.append(noised_title)
                suffixes.append(suffix)
                noised_suffixes.append(noised_suffix)

            total_loss += self.train_character_classifier(
                full_noised_names, char_classes).item()
            total_loss += self.train_aux_classifier(
                self.title_classifier, noised_titles, titles).item()
            total_loss += self.train_aux_classifier(
                self.suffix_classifier, noised_suffixes, suffixes).item()
            total_loss += self.train_DAE(self.first_DAE,
                                         noised_firsts, firsts).item()
            total_loss += self.train_DAE(self.first_DAE,
                                         noised_middle_initials, middle_initials).item()
            total_loss += self.train_DAE(self.last_DAE,
                                         noised_lasts, lasts).item()

            if i % save_every == 0:
                self.save_checkpoint()

            if i % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0
                self.plot_losses(
                    all_losses, f'Iteration of {batch_sz} batch size', 'Cross Entropy Loss Sum')

        self.save_checkpoint()

    def train_character_classifier(self, src: list, trg: list):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.character_classifier.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss = 0

        max_len = len(max(src, key=len))
        in_vocab = self.character_classifier.input
        out_vocab = self.character_classifier.output

        input = torch.LongTensor(convertToIdxList(
            src, in_vocab, max_len)).transpose(0, 1).to(DEVICE)
        length_lst = [len(name) for name in src]
        length_tnsr = torch.LongTensor(length_lst).to(DEVICE)
        trg = torch.LongTensor(convertToIdxList(
            trg, out_vocab, max_len)).transpose(0, 1).to(DEVICE)

        outputs, hidden = self.character_classifier.encode(input, length_tnsr)

        for i in range(max_len):
            probs = self.character_classifier.decode(outputs[i])
            loss += criterion(probs, trg[i])

        loss.backward()
        optimizer.step()

        return loss

    def train_aux_classifier(self, classifier: AuxClassifier, src: list, trg: list) -> int:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            classifier.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss = 0

        max_src_len = len(max(src, key=len))
        in_vocab = classifier.input
        out_vocab = classifier.output

        input = torch.LongTensor(convertToIdxList(
            src, in_vocab, max_src_len)).transpose(0, 1).to(DEVICE)
        len_input = torch.LongTensor([len(name) for name in src]).to(DEVICE)
        trg = torch.LongTensor([out_vocab.index(title)
                                for title in trg]).to(DEVICE)

        output = classifier.forward(input, len_input)

        loss += criterion(output, trg)
        loss.backward()
        optimizer.step()

        return loss

    def train_DAE(self, dae: DenoisingAutoEncoder, src: list, trg: list) -> int:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(dae.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss = 0

        max_src_len = len(max(src, key=len))
        max_trg_len = len(max(trg, key=len))
        in_vocab = dae.input
        out_vocab = dae.output

        # Encoder inputs
        encoder_in = torch.LongTensor(convertToIdxList(
            src, in_vocab, max_src_len)).transpose(0, 1).to(DEVICE)
        len_input = torch.LongTensor([len(name) for name in src]).to(DEVICE)

        # Decoder inputs and target
        decoder_in = torch.LongTensor(convertToIdxList(
            trg, out_vocab, max_trg_len, w_start=True)).transpose(0, 1).to(DEVICE)
        trg_tnsr = torch.LongTensor(convertToIdxList(
            trg, out_vocab, max_trg_len, w_end=True)).transpose(0, 1).to(DEVICE)

        _, hidden = dae.encode(encoder_in, len_input)

        # Should be max_trg_len + 1 for SOS and EOS
        for i in range(max_trg_len + 1):
            output, hidden = dae.forward(decoder_in[i].unsqueeze(0), hidden)
            loss += criterion(output[0], trg_tnsr[i])

        loss.backward()
        optimizer.step()

        return loss

    def test(self, df: pandas.DataFrame):
        return False

    def test_name(self, name):
        classification_lst = self.test_character_classifier([name])

        title, firsts, middles, lasts, suffix = parse_name(
            name, classification_lst)

        cleaned_firsts = self.test_DAE(
            self.first_DAE, firsts) if len(firsts) > 0 else None
        cleaned_middles = self.test_DAE(
            self.first_DAE, middles) if len(middles) > 0 else None
        cleaned_lasts = self.test_DAE(
            self.last_DAE, lasts) if len(lasts) > 0 else None
        cleaned_title = self.test_aux_classifier(
            self.title_classifier, title) if len(title) > 0 else None
        cleaned_suffix = self.test_aux_classifier(
            self.suffix_classifier, suffix) if len(suffix) > 0 else None

        return {'first': cleaned_firsts, 'middle': cleaned_middles, 'last': cleaned_lasts, 'title': cleaned_title, 'suffix': cleaned_suffix}

    def test_character_classifier(self, inputs: list) -> list:
        batch_sz = len(inputs)
        max_len = len(max(inputs, key=len))
        in_vocab = self.character_classifier.input
        out_vocab = self.character_classifier.output

        input = torch.LongTensor(convertToIdxList(
            inputs, in_vocab, max_len)).transpose(0, 1).to(DEVICE)
        length_lst = [len(name) for name in inputs]
        length_tnsr = torch.LongTensor(length_lst).to(DEVICE)

        outputs, hidden = self.character_classifier.encode(input, length_tnsr)

        classifications = []

        for i in range(max_len):
            probs = self.character_classifier.decode(outputs[i])
            _, idxes = probs.max(1)
            classifications.extend([out_vocab[idxes[j].item()]
                                    for j in range(len(idxes))])

        return classifications

    def test_aux_classifier(self, classifier: AuxClassifier, inputs: list) -> list:
        batch_sz = len(inputs)
        max_src_len = len(max(inputs, key=len))
        in_vocab = classifier.input
        out_vocab = classifier.output

        input = torch.LongTensor(convertToIdxList(
            inputs, in_vocab, max_src_len)).transpose(0, 1).to(DEVICE)
        len_input = torch.LongTensor([len(name) for name in inputs]).to(DEVICE)

        output = classifier.forward(input, len_input)
        _, max_idxes = output.max(1)

        return [out_vocab[max_idxes[i].item()] for i in len(max_idxes)]

    def test_DAE(self, dae: DenoisingAutoEncoder, inputs: list) -> list:
        batch_sz = len(inputs)
        max_len = len(max(inputs, key=len))
        in_vocab = dae.input
        out_vocab = dae.output

        # Encoder inputs
        encoder_in = torch.LongTensor(convertToIdxList(
            inputs, in_vocab, max_len)).transpose(0, 1).to(DEVICE)
        len_input = torch.LongTensor([len(name) for name in inputs]).to(DEVICE)

        _, hidden = dae.encode(encoder_in, len_input)
        input = torch.LongTensor(
            [in_vocab.index(SOS)] * batch_sz).unsqueeze(0).to(DEVICE)
        # Should be max_trg_len + 1 for SOS and EOS

        all_EOS_or_PAD = False
        end_of_seq_signals = [out_vocab.index(EOS), out_vocab.index(PAD)]
        cleaned_names = []

        while not all_EOS_or_PAD:
            output, hidden = dae.forward(input, hidden)
            _, input = output.max(2)
            cleaned_names.extend([out_vocab[input[i]]
                                  for i in range(len(input))])
            all_EOS_or_PAD = doesTensorOnlyHasValues(input, end_of_seq_signals)

        return cleaned_names

    def save_checkpoint(self, folder: str = 'Weights'):
        dae_fn_fp = os.path.join(folder, f'{self.session_name}_fn_dae')
        dae_ln_fp = os.path.join(folder, f'{self.session_name}_ln_dae')
        aux_fp = os.path.join(folder, f'{self.session_name}_auxilary')
        classifier_fp = os.path.join(folder, f'{self.session_name}_classifier')

        if not os.path.exists(folder):
            os.mkdir(folder)

        fn_dae_content = {'weight': self.first_DAE.state_dict()}
        ln_dae_content = {'weight': self.last_DAE.state_dict()}
        aux_content = {'title_weight': self.title_classifier.state_dict(),
                       'suffix_weight': self.suffix_classifier.state_dict()}
        classifier_content = {
            'classifier': self.character_classifier.state_dict()}

        torch.save(fn_dae_content, dae_fn_fp)
        torch.save(ln_dae_content, dae_ln_fp)
        torch.save(aux_content, aux_fp)
        torch.save(classifier_content, classifier_fp)

    def load_checkpoint(self, name: str = None, folder: str = 'Weights'):
        if name is None:
            name = self.session_name

        fn_dae_fp = os.path.join(folder, f'{name}_fn_dae')
        ln_dae_fp = os.path.join(folder, f'{name}_ln_dae')
        aux_fp = os.path.join(folder, f'{name}_auxilary')
        classifier_fp = os.path.join(folder, f'{name}_classifier')

        fn_dae_content = torch.load(fn_dae_fp, map_location=DEVICE)
        ln_dae_content = torch.load(ln_dae_fp, map_location=DEVICE)
        aux_content = torch.load(aux_fp, map_location=DEVICE)
        classifier_content = torch.load(classifier_fp, map_location=DEVICE)

        self.first_DAE.load_state_dict(fn_dae_content['weight'])
        self.last_DAE.load_state_dict(ln_dae_content['weight'])
        self.title_classifier.load_state_dict(aux_content['title_weight'])
        self.suffix_classifier.load_state_dict(aux_content['suffix_weight'])
        self.character_classifier.load_state_dict(
            classifier_content['classifier'])

    def plot_losses(self, loss: list, x_label: str, y_label: str, folder: str = "Plot"):
        x = list(range(len(loss)))
        plt.plot(x, loss, 'r--', label="Loss")
        plt.title("Losses")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper left')
        plt.savefig(f"{folder}/{self.session_name}")
        plt.close()
