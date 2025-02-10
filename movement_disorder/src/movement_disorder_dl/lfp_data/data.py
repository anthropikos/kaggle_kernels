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
from torch.utils.data import Dataset, DataLoader, IterableDataset
from collections import namedtuple

data_dir = Path("../data/essential_tremor")

# For each patient get either pegboard/posture/pouring and on/off state
class EssentialTremorPatientDataset():
    def __init__(self, patient_num, data_dir=data_dir):
        assert isinstance(patient_num, int) & (patient_num in range(1, 9)), f"'patient_num' has to be an int within the range of 1-8 inclusive."
        self.patient_num = patient_num
        self.data_dir = data_dir
        self.patient_data_dir = self.data_dir / Path(f"ET{patient_num}")
        self.activities = None
        self.states = None
        # self._files = sorted( self.patient_data_dir.glob("*.h5") )
        # assert len(self._files) == 6, f"There should be 6x HDF5 files in {self.patient_data_dir}, but found {len(self._files)}."

    def __getitem__(self, key:tuple[str, str]):
        Output = namedtuple("Patient_LFP_Data", ["patient_num", "activity", "state", "LFP", "label"])
        
        assert (isinstance(key, tuple) & (len(key) == 2) & isinstance(key[0], str) & isinstance(key[1], str), 
                f"'key' has a be a length 2 tuple of strings indicating the activity and state."
                )
        activity, state = key
        activity, state = activity.lower(), state.lower()
        assert activity in ["posture", "pouring"], f"First element of tuple (i.e., activity) expect 'posture' or 'pouring', got {activity}."
        assert state in ["on", "off"], f"Second element of tuple (i.e., state) expect 'on' or 'off', got {state}."

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
            
    # def __len__(self):
        # return len(self._files)


# Wrapper for all the patients
class EssentialTremorLFPDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, key):
        pass

    def __len__(self):
        pass