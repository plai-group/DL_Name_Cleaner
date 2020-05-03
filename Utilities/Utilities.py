import torch
from Constant import *


def convertToIdxList(data: list, vocab: list, max_len: int, eos_on: bool = False):
    ret = []

    for point in data:
        diff = max_len - len(point)
        end = [vocab.index(EOS)] + [vocab.index(PAD)] * \
            (diff - 1) if eos_on else [vocab.index(PAD)] * diff
        ret.append([vocab.index(c) for c in point] + end)

    return ret
