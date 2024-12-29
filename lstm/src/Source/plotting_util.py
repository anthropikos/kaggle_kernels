## Plotting utilities
## Anthony Lee 2024-12-20

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_train_validation_loss(avg_training_loss: np.ndarray, avg_validation_loss: np.ndarray) -> mpl.axes.Axes:
    assert len(avg_training_loss) == len(
        avg_validation_loss
    ), f"Training loss and validation loss arrays should have the same length, got {len(avg_training_loss)} and {len(avg_validation_loss)}"

    fig, ax = plt.subplots()
    marker_size = 10

    ax.set_title("Avg training and validation loss for each epoch")
    ax.set_ylabel("Avg [training | validation] loss")
    ax.set_xlabel("Epoch")

    num_of_epochs = len(avg_training_loss)

    ax.scatter(
        range(num_of_epochs),
        avg_training_loss,
        color="C1",
        s=marker_size,
        label="avg training loss",
    )
    ax.scatter(
        range(num_of_epochs),
        avg_validation_loss,
        color="C2",
        s=marker_size,
        label="avg validation loss",
    )

    ax.plot(avg_training_loss, "--", alpha=0.3, color="C1")
    ax.plot(avg_validation_loss, "--", alpha=0.3, color="C2")

    ax.legend(loc="upper right")
    ax.grid(visible=True, which="both", axis="both", alpha=0.2)
    return ax
