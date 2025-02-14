# 2025-02-12 Anthony Lee

from pathlib import Path
from src import movement_disorder_dl as md
from torch.utils.data import DataLoader
# import lightning as L

def main():
    data_dir = Path("data/essential_tremor")
    dataset = md.lfp_data.EssentialTremorLFPDataset_Posture(data_dir=data_dir)
    dataloader = DataLoader(dataset=dataset, batch_size=10)
    model = md.model.CNN1d()


    for idx, batch in enumerate(dataloader):
        if idx > 5: break

        input, label = batch

        output = model(input)

        






if __name__ == "__main__":
    main()