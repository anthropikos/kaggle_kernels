## Utilities for Kaggle related stuff
## Anthony Lee 2024-12-20
from typing import Iterable
import torch
import pandas as pd
from .data_processing_util import text_to_vector

def predict_test_data_for_submission(model: torch.nn.Module, df_test: pd.DataFrame, save: bool = None) -> Iterable:
    """Convenience function to predict for submission."""
    if save is None:
        save = False

    test_datas_vectorized = text_to_vector(df_test.text.to_list())

    model.train(False)
    holder = []

    for item in test_datas_vectorized:
        prediction = model(torch.tensor(item)).detach()

        # Convert probability to categorical label
        if prediction > 0.5:
            prediction = 1
        else:
            prediction = 0

        holder.append(prediction)

    if save is True:
        submission = pd.DataFrame({"id": df_test.id, "target": holder})
        submission.to_csv("/kaggle/working/submission.csv", index=False)

    return holder