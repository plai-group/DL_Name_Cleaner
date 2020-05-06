import string
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS = 'SOS'  # '<SOS>'
EOS = 'EOS'  # '<EOS>'
PAD = 'PAD'  # '<PAD>'
AUX_CHARS = [SOS, PAD, EOS]

INPUT = [c for c in string.printable] + AUX_CHARS
INPUT_SZ = len(INPUT)
DAE_OUTPUT = [c for c in string.ascii_letters] + AUX_CHARS
DAE_OUTPUT_SZ = len(DAE_OUTPUT)

TITLES = ['Mr', 'Ms', 'Dr', 'Mrs', 'Sir', "Ma'am", 'Madam']
SUFFIXES = ['Sr', 'Snr', 'Jr', 'Jnr', 'PhD', 'MD', 'I', 'II', 'III', 'IV']
NAME_FORMATS = ['{f} {l}', '{f} {m} {l}', '{l}, {f}', '{l}, {f} {m}']
CHARACTER_CLASSIFICATIONS = ['t', 'f', 'm', 'l',
                             's', '.', ' ', ',', '-', '\'', EOS, PAD]
