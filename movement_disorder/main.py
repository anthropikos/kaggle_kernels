# 2025-02-12 Anthony Lee

import logging
from argparse import ArgumentParser
from movement_disorder_dl import scripts
import warnings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings('ignore')  # TODO: (Later) Fix torch warning on zero-padded copy of input - /opt/homebrew/Caskroom/miniconda/base/envs/movement_disorder/lib/python3.12/site-packages/torch/nn/modules/conv.py:370: UserWarning: Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created (Triggered internally at /Users/runner/miniforge3/conda-bld/libtorch_1741562946353/work/aten/src/ATen/native/Convolution.cpp:1037.)

parser = ArgumentParser(
    prog='movement_disorder_deep_learning_model',
    description='Trains and inferences on LFP data of Parkinsons and Essential Tremor.',
)

parser.add_argument('mode')

args = parser.parse_args()

if __name__ == "__main__":
    match args.mode:
        case 'train':
            scripts.training.train_model()
        case 'tune':
            scripts.tuning.tune_model()
        case 'inference':
            results = scripts.inferencing.inference()
            print(f'Probability of in Tremor state: {results}')
        case _:
            raise ValueError(f'First positional argument `mode` has to be either `train`, `tune` or `inference`, got {args.mode}')