# 2025-02-12 Anthony Lee

# from pathlib import Path
# from src import movement_disorder_dl as md
# from torch.utils.data import DataLoader
# # import lightning as L
# 
# def main():
#     data_dir = Path("data/essential_tremor")
#     dataset = md.lfp_data.EssentialTremorLFPDataset_Posture(data_dir=data_dir)
#     dataloader = DataLoader(dataset=dataset, batch_size=10)
#     model = md.model.CNN1d()
# 
# 
#     for idx, batch in enumerate(dataloader):
#         if idx > 5: break
# 
#         input, label = batch
# 
#         output = model(input)

from src.movement_disorder_dl.model import CNN1d
from src.movement_disorder_dl.lfp_data import EssentialTremorLFPDataset_Posture
from torch.utils.data import DataLoader

def main():
    dataset = EssentialTremorLFPDataset_Posture()

    print(dataset[0])

if __name__ == "__main__":
    main()
