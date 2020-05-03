import torch
from Constant import *


def convertToIdxList(data: list, vocab: list, max_len: int, w_start: bool = False, w_end: bool = False):
    ret = []

    for point in data:
        diff = max_len - len(point)
        pads = [vocab.index(PAD)] * diff

        if w_start and w_end:
            ret.append([vocab.index(SOS)] + [vocab.index(c)
                                             for c in point] + [vocab.index(EOS)] + pads)
        elif w_start:
            ret.append([vocab.index(SOS)] + [vocab.index(c)
                                             for c in point] + pads)
        elif w_end:
            ret.append([vocab.index(c)
                        for c in point] + [vocab.index(EOS)] + pads)
        else:
            ret.append([vocab.index(c) for c in point] + pads)

    return ret
