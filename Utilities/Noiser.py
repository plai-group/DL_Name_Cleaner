import math
import string
import torch

CHARACTER_REPLACEMENT = dict()
CHARACTER_REPLACEMENT['A'] = 'QSZWXaU'
CHARACTER_REPLACEMENT['B'] = 'NHGVb'
CHARACTER_REPLACEMENT['C'] = 'VFDXc'
CHARACTER_REPLACEMENT['D'] = 'FRESXCd'
CHARACTER_REPLACEMENT['E'] = 'SDFR$3#WSe'
CHARACTER_REPLACEMENT['F'] = 'GTRDCVf'
CHARACTER_REPLACEMENT['G'] = 'HYTFVBg'
CHARACTER_REPLACEMENT['H'] = 'JUYTGBNh'
CHARACTER_REPLACEMENT['I'] = 'UJKLO(*i'
CHARACTER_REPLACEMENT['J'] = 'MKIUYHNj'
CHARACTER_REPLACEMENT['K'] = 'JM<LOIk'
CHARACTER_REPLACEMENT['L'] = 'K<>:POl'
CHARACTER_REPLACEMENT['M'] = 'NJK<m'
CHARACTER_REPLACEMENT['N'] = 'BHJMn'
CHARACTER_REPLACEMENT['O'] = 'PLKI()Po'
CHARACTER_REPLACEMENT['P'] = 'OL:{_)O"p'
CHARACTER_REPLACEMENT['Q'] = 'ASW@!q'
CHARACTER_REPLACEMENT['R'] = 'TFDE$r%'
CHARACTER_REPLACEMENT['S'] = 'DXZAWEs'
CHARACTER_REPLACEMENT['T'] = 'YGFR%^t'
CHARACTER_REPLACEMENT['U'] = 'IJHY&*uA'
CHARACTER_REPLACEMENT['V'] = 'CFGBvW'
CHARACTER_REPLACEMENT['W'] = 'SAQ@#EVw'
CHARACTER_REPLACEMENT['X'] = 'ZASDCx'
CHARACTER_REPLACEMENT['Y'] = 'UGHT^&y'
CHARACTER_REPLACEMENT['Z'] = 'XSAz'
CHARACTER_REPLACEMENT['a'] = 'qwszAu'
CHARACTER_REPLACEMENT['b'] = 'nhgvBp'
CHARACTER_REPLACEMENT['c'] = 'vfdxC'
CHARACTER_REPLACEMENT['d'] = 'fresxcD'
CHARACTER_REPLACEMENT['e'] = 'sdfr43wsE'
CHARACTER_REPLACEMENT['f'] = 'gtrdcvFy'
CHARACTER_REPLACEMENT['g'] = 'hytfvbG'
CHARACTER_REPLACEMENT['h'] = 'juytgbnH'
CHARACTER_REPLACEMENT['i'] = 'ujklo;98I'
CHARACTER_REPLACEMENT['j'] = 'mkiuyhnJ'
CHARACTER_REPLACEMENT['k'] = 'jm,loijK'
CHARACTER_REPLACEMENT['l'] = 'k,.;pokL'
CHARACTER_REPLACEMENT['m'] = 'njk,M'
CHARACTER_REPLACEMENT['n'] = 'bhjmN'
CHARACTER_REPLACEMENT['o'] = 'plki90pO'
CHARACTER_REPLACEMENT['p'] = 'ol;[-b0oP'
CHARACTER_REPLACEMENT['q'] = 'asw21bQ'
CHARACTER_REPLACEMENT['r'] = 'tfde45R'
CHARACTER_REPLACEMENT['s'] = 'dxzawe5S'
CHARACTER_REPLACEMENT['t'] = 'ygfr56T'
CHARACTER_REPLACEMENT['u'] = 'ijhy78Ua'
CHARACTER_REPLACEMENT['v'] = 'cfgbVw'
CHARACTER_REPLACEMENT['w'] = 'saq23evW'
CHARACTER_REPLACEMENT['x'] = 'zsdcX'
CHARACTER_REPLACEMENT['y'] = 'uhgt67Y'
CHARACTER_REPLACEMENT['z'] = 'xsaNZ'
CHARACTER_REPLACEMENT['-'] = '_=+~'
CHARACTER_REPLACEMENT['.'] = ',\';`'
CHARACTER_REPLACEMENT['\''] = '"`'


def noise_name(name: str, allowed_noise_chars: list, noise_prob: float):
    name_length = len(name)
    noise_ret = []
    noised_name = ''

    # [no noise, add character, switch with similar, remove]
    if name_length == 1:
        noise_dist = torch.tensor(
            [1 - noise_prob, (noise_prob / 2), (noise_prob / 2), 0, 0])
    else:
        noise_dist = torch.tensor([1 - noise_prob] + [noise_prob / 3] * 3)

    for i in range(name_length):
        current_char = name[i]

        sample = int(torch.distributions.Categorical(
            noise_dist).sample().item())

        if sample == 0:
            noised_name = noised_name + current_char
        elif sample == 1:
            noise_chars_len = len(allowed_noise_chars)
            sampled_idx = int(torch.distributions.Categorical(torch.FloatTensor(
                [1 / noise_chars_len] * noise_chars_len)).sample().item())
            sampled_char = allowed_noise_chars[sampled_idx]
            noised_name = noised_name + current_char + sampled_char
        elif sample == 2:
            replacements = CHARACTER_REPLACEMENT[current_char]
            replacements_len = len(replacements)
            sampled_idx = int(torch.distributions.Categorical(
                torch.FloatTensor([1/replacements_len] * replacements_len)).sample().int())
            sampled_char = replacements[sampled_idx]
            noised_name = noised_name + sampled_char

    return noised_name
