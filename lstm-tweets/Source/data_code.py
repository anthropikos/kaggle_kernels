## Some data utility code
## Anthony Lee 2024-12-17

from typing import List, Union, Iterable, Tuple, Dict
from multiprocessing import set_start_method, cpu_count
from pathlib import Path
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import spacy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from spacy.cli import download  # Download SpaCy model  Referenece: https://stackoverflow.com/questions/69304467/how-to-download-en-core-web-sm-model-at-runtime-in-spacy

def spacy_setup(language_model:str = "en_core_web_lg"):
    """Setup the SpaCy tool."""
    download(language_model)

def read_in_csv(data_dir_path: Union[Path, str] = None) -> Tuple:
    """Read in the dataset CSVs and return a tuple of dataframes."""

    if data_dir_path is None: 
        data_dir_path = f"../input"
        data_dir_path = Path(data_dir_path).resolve().absolute()
    else:
        data_dir_path = Path(data_dir_path).resolve().absolute()

    ## Read in the data
    df_train = pd.read_csv(data_dir_path / Path("nlp-getting-started/train.csv"))
    df_test = pd.read_csv(data_dir_path / Path("nlp-getting-started/test.csv"))
    df_sample_submission = pd.read_csv(data_dir_path / Path("nlp-getting-started/sample_submission.csv"))

    OutputTuple = namedtuple("train_test_example", ["train", "test", "submission_example"])
    return OutputTuple(train=df_train, test=df_test, submission_example=df_sample_submission)


def train_validation_split(df_train: pd.DataFrame, validation_fraction: float = 0.1) -> Tuple:
    """Train/Validation split the train dataframe."""

    ## Simple Train/Validate split
    train_datas, validation_datas, train_targets, validation_targets = train_test_split(
        df_train.text.to_list(),
        df_train.target.to_list(),
        train_size=(1 - validation_fraction),
        random_state=7,  # For consistency
        stratify=df_train.target.to_list(),
    )

    OutputTuple = namedtuple(
        "train_validation_split",
        ["train_datas", "validation_datas", "train_targets", "validation_targets"],
    )
    return OutputTuple(
        train_datas=train_datas,
        validation_datas=validation_datas,
        train_targets=train_targets,
        validation_targets=validation_targets,
    )


def text_to_vector(documents: Iterable, n_process=None) -> List:
    """Iterate through a list of documents and returns a list of vectorized documents.

    Uses spaCy's en_core_web_log model to transform each tweet into a 2D ndarray
    of floats. Each ndarray is the word_count by 300 where 300 is the vector length
    of each token used in the spaCy model.
    """
    language_model = "en_core_web_lg"
    
    if not isinstance(documents, Iterable):
        raise TypeError("`documents` has to be a list of strings.")

    nlp = spacy.load(language_model)

    vec_length = 300  # Token vector length in SpaCy

    if n_process is None:
        n_process = cpu_count()

    docs = nlp.pipe(texts=documents, n_process=n_process, batch_size=50)

    holder_all_tweets = []

    for doc in docs:
        tweet_length = len(doc)
        doc_ndarray = np.zeros(shape=(tweet_length, vec_length), dtype=np.float64)

        for idx, token in enumerate(doc):
            doc_ndarray[idx, :] = token.vector
        holder_all_tweets.append(doc_ndarray)

    return holder_all_tweets


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


def load_dataset(mode:str="all", data_directory:Union[str, Path]=None) -> tuple:
    """Data processing pipeline and return dataset objects for both train and validation sets."""
    
    if data_directory is not None:
        data_directory = Path(data_directory)

    # Read in the text files
    dfs = read_in_csv(data_dir_path=data_directory)
    df_train = dfs.train
    train_data, validation_data, train_target, validation_target = train_validation_split(
        df_train=df_train, validation_fraction=0.2
    )

    # Tokenize and encode / vectorize
    train_data = text_to_vector(train_data)
    validation_data = text_to_vector(validation_data)

    # Create dataset objects
    train_dataset = DisasterTweetDataset(train_data, train_target)
    validation_dataset = DisasterTweetDataset(validation_data, validation_target)

    DatasetOutput = namedtuple("train_validation_datasets", ["train_dataset", "validation_dataset"])
    return DatasetOutput(train_dataset=train_dataset, validation_dataset=validation_dataset)