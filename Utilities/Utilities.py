import torch
from Constant import *


def convertToIdxList(data: list, vocab: list, max_len: int):
    ret = []

    for point in data:
        diff = max_len - len(point)
        end = [vocab.index(PAD)] * diff
        ret.append([vocab.index(c) for c in point] + end)

    return ret


def convertToIdxListWStartEnd(data: list, vocab: list, max_len: int):
    ret = []

    for point in data:
        diff = max_len - len(point) + 2
        end = [vocab.index(EOS)] + [vocab.index(PAD)] * (diff - 1)
        ret.append([vocab.index(SOS)] + [vocab.index(c) for c in point] + end)

    return ret
