## Dataset
## Anthony Lee 2024-12-20

from torch.utils.data import Dataset
from typing import Iterable, Tuple, List, Union
from collections import namedtuple
from pathlib import Path
from .data_processing_util import text_to_vector, read_in_csv, train_validation_split

class DisasterTweetDataset(Dataset):
    def __init__(self, vectorized_tweets: Iterable, targets: Iterable) -> None:
        self.vectorized_tweets = vectorized_tweets
        self.targets = targets

        self.__data_validation()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[List, List]:
        ReturnedResult = namedtuple("disaster_tweet", ["target", "vectorized_tweet"])

        target = self.targets[idx]
        vectorized_tweet = self.vectorized_tweets[idx]

        return ReturnedResult(target=target, vectorized_tweet=vectorized_tweet)

    def __data_validation(self) -> None:
        if len(self.vectorized_tweets) != len(self.targets):
            raise ValueError(f"The data counts do NOT match, got {len(self.vectorized_tweets)} and {len(self.targets)}")