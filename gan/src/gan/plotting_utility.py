## Plotting utilities
## Anthony Lee 2024-12-23

import matplotlib as mpl
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np


def image_preview(dataset: Dataset, row: int, col: int) -> mpl.figure.Figure:
    fig, axs = plt.subplots(row, col, figsize=(10, 10))
    axs = axs.flatten()

    for i in range(row * col):
        idx = np.random.randint(low=0, high=len(dataset))
        img = dataset[idx]
        axs[i].imshow(img)
        axs[i].set_axis_off()

    fig.tight_layout()
    return fig
