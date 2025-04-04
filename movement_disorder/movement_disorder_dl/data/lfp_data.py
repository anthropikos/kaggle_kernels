# Anthony Lee 2025-02-07

# >>> Code information <<< GNU AGPLv3
# Code Owner: Anthony Lee

# >>> Dataset information <<< Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
# Data Owner: Rodriguez F, He S, Tan H, and the Tan Group
# Dataset Name: Local Field Potential (LFP) data recorded from externalized Essential Tremor DBS Patients during 3 upper-limb movement tasks
# DOI: 10.5287/bodleian:ZVNyvrw7R

import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset
from collections import namedtuple
from tqdm import tqdm


# DATA_DIR = Path("../data/essential_tremor").resolve()  # This doesn't work because it would be relative to where the interpreter is opened.
DATA_DIR = Path(__file__).parent / Path("../../data/essential_tremor") # A hack and not ideal  TODO: Fix this data dir hack.
PATIENT_NUM_RANGE = range(1, 9)
SAMPLING_RATE = 2048  # 2048 Hz according to the dataset
LABEL_ON_THRESHOLD = 0.25 # Percentage of number of YES labels in window to be considered pathological

# TODO: Think of another way to resolve this data dir issue...
assert DATA_DIR.is_dir(), '`DATA_DIR` is not a directory, file path may be wrong.'

# For each patient get either pegboard/posture/pouring and on/off state
class EssentialTremorPatientDataset():
    def __init__(self, patient_num, data_dir=DATA_DIR):
        assert isinstance(patient_num, int) & (patient_num in PATIENT_NUM_RANGE), f"'patient_num' has to be an int within the range of 1-8 inclusive."
        self.patient_num = patient_num
        self.data_dir = data_dir
        self.patient_data_dir = self.data_dir / Path(f"ET{patient_num}")
        self.__hdf5_files = self.patient_data_dir.glob("*.h5")
        self.activities, self.states = self.__find_set_of_activities_and_states()

    def __getitem__(self, key:tuple[str, str]):
        Output = namedtuple("Patient_LFP_Data", ["patient_num", "activity", "state", "LFP", "label"])

        assert (isinstance(key, tuple) & (len(key) == 2)), f"`key` is expected to be of type `tuple` and length 2, got type {type(key)} and length {len(key)}."
        assert (isinstance(key[0], str) & isinstance(key[1], str)), f"Elements of 'key' have to be of type `str`, got ({type(key[0])}, {type(key[1])})."
        activity, state = key[0].lower(), key[1].lower()
        assert activity in self.activities, f"First element of tuple (i.e., activity) expect one of the followings {self.activities}, got {activity}."
        assert state in self.states, f"Second element of tuple (i.e., state) expect one of the followings {self.states}, got {state}."

        filepath = self.patient_data_dir / Path(f"{activity.capitalize()}_{state}.h5")
        
        with h5py.File(filepath, "r") as h5:
            data_lfp = h5["LFP"][:]
            data_label = h5["label"][:]

        return Output(
            patient_num=self.patient_num,
            activity=activity,
            state=state,
            LFP=data_lfp,
            label=data_label,
        )
            
    def __len__(self):
        return len(self.__hdf5_files)

    def __find_set_of_activities_and_states(self):
        activities, states = [], []
        for item in self.__hdf5_files:
            activity, state = item.stem.split("_")
            activities.append(activity.lower())
            states.append(state.lower())

        activities = set(activities)
        states = set(states)
        
        assert len(activities) != 0, '`activities` is empty.'
        assert len(states) != 0, '`states` is empty'

        return activities, states


# Wrapper for all the patients' posture dataset
class EssentialTremorLFPDataset_Posture(Dataset):
    """Only looking at the Posture data because all patients have this activity type."""
    def __init__(self, data_dir=DATA_DIR):
        self.holder_lfp = []
        self.holder_label = []  # Reference to torch.tensor objects

        for patient_num in tqdm(PATIENT_NUM_RANGE, desc="Parsing data...",):
            # print(f"Parsing patient {patient_num} hdf5 files...")
            dataset = EssentialTremorPatientDataset(patient_num=patient_num, data_dir=data_dir)
            data_off = dataset["posture", "off"]

            (_, _, _, lfp_on, label_on) = dataset["posture", "on"]
            
            # LFP data - Sliding window with memoryview - Torch views avoid explicit copies of the data
            lfp_on = lfp_on.mean(axis=0).squeeze()  # Average LFP across all DBS channels
            lfp_on = torch.tensor(lfp_on, dtype=torch.float32).unfold(0, SAMPLING_RATE, 1)
            
            # Label data - Torch views avoid explicit copies of the data
            label_on = torch.tensor(label_on, dtype=torch.float32).unfold(0, SAMPLING_RATE, 1).mean(dim=1, keepdim=False)
            
            self.holder_lfp.append(lfp_on)
            self.holder_label.append(label_on)
        
        # ISSUE >>> Concat turns the view into concrete tensors and thus takes up too much memory
        # self.lfp = torch.concat(holder_lfp, dim=0)
        # self.label = torch.concat(holder_label, dim=0)

        # Create range mapping for each of the tensor
        endpoints = [tensor.shape[0] for tensor in self.holder_lfp]
        cumsum = torch.tensor([0]+endpoints).cumsum(dim=0)
        ranges = cumsum.unfold(0, 2, 1)
        self.range_map = {idx:torch.arange(start=start, end=end) for idx, (start, end) in enumerate(ranges)}
        return

    def __getitem__(self, idx):
        if (idx < 0) | (idx >= len(self)):
            raise IndexError(f"Index out of range, expect index to be between 0-{len(self)-1} inclusive.")
        list_idx, item_idx = self.__determine_list_and_item_idx(idx)
        
        lfp = self.holder_lfp[list_idx][item_idx].reshape(shape=(1, -1))            # 1x channel dimension
        label = self.holder_label[list_idx][item_idx].unsqueeze(0)                  # Add new dimension, returns a view
        # label = self.holder_label[list_idx][item_idx][None]                       # Add new dimension, returns a view

        return lfp, label
        
    def __len__(self):
        last_key = PATIENT_NUM_RANGE[-1] - 1  # Zero index
        last_range = self.range_map[last_key]
        return last_range[-1]

    def __determine_list_and_item_idx(self, idx):
        """Determine which item of the list to extract data from."""
        for list_idx, value in self.range_map.items():
            if idx in value:
                if list_idx == 0:
                    item_idx = idx
                    return list_idx, item_idx
                else:
                    previous_range_end = self.range_map[list_idx - 1][-1]
                    item_idx = idx - previous_range_end - 1
                    return list_idx, item_idx