import torch
import argparse
from Pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session',
                    nargs='?', default='UNNAMED', type=str)
parser.add_argument('--hidden_size', help='Size of RNN hidden layers',
                    nargs='?', default=256, type=int)
parser.add_argument(
    '--num_layers', help='Hidden size for format models', nargs='?', default=6, type=int)

args = parser.parse_args()
NAME = args.name
HIDDEN_SZ = args.hidden_size
NUM_LAYERS = args.num_layers

pipeline = Pipeline(NAME, HIDDEN_SZ, NUM_LAYERS)
pipeline.load_checkpoint()
pipeline.test_name('Jason')
