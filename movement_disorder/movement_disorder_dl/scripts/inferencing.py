from ..model.cnn_1d_lightning import CNN1d_Lightning
from pathlib import Path
import torch
import numpy as np
from .. import config
import h5py

# TODO: (Later) Test whether the model is just predicting all to be true/false

##### Manual parsing an example input #####
def find_idx_of_switches(labels:list) -> list:
    idx_switch = np.argwhere(np.diff(labels) != 0).squeeze()
    return idx_switch

def find_true_false_segments(labels, debug:bool=None) -> tuple:
    """Given labels, return the idx ranges of True-/False-segments."""
    if debug is None:
        debug = False
        
    true_segments = []
    false_segments = []

    idx_switch = find_idx_of_switches(labels=labels)

    segments = np.lib.stride_tricks.sliding_window_view(idx_switch, window_shape=2)
    segments = np.concat( [[[-1, segments[0][0]]], segments], axis=0 )  # Add the very first segment that doesn't have the initial switch to bookend - Negative 1 because of later offset

    
    for beginning, end in segments: 
        beginning += 1  # Because of how np.diff works, the index marks the element before the start of segment
        
        if debug: 
            print(label[beginning], label[end])
            print(np.mean(label[beginning: end+1], dtype=int))
        
        assert (label[beginning] == label[end]) & (label[end] == np.mean(label[beginning:end+1], dtype=int))
        
        match bool(label[beginning]):  # np.bool vs regular Python bool
            case True:
                true_segments.append( (beginning, end) )
            case False:
                false_segments.append( (beginning, end) )
            case _:
                raise ValueError(f'Received {label[beginning]} of type {type(label[beginning])}')
                
    return (true_segments, false_segments)

def get_lfp_and_label_of_posture_on(patient_num:int):
    
    filepath = config.LaptopConfig().DATA_DIR_PATH / Path(f'ET{patient_num}/Posture_on.h5')

    with h5py.File(filepath, 'r') as file: 
        lfp_data = file['LFP'][:]
        lfp_label = file['label'][:]
        
    return lfp_data, lfp_label

lfp, label = get_lfp_and_label_of_posture_on(patient_num=5)
true_segs, false_segs = find_true_false_segments(labels=label)

true_seg, false_seg = true_segs[0], false_segs[0]

true_lfp = lfp[:, true_seg[0]:true_seg[1]]
false_lfp = lfp[:, false_seg[0]:false_seg[1]]


input = torch.tensor(true_lfp).mean(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).float()


def inference(input_path=None, checkpoint_path=None):
    if checkpoint_path is None: 
        checkpoint_path = Path(__file__).parent / Path('../../lightning_logs/version_0/checkpoints/epoch=8-step=87291.ckpt')
    model = CNN1d_Lightning.load_from_checkpoint(checkpoint_path=checkpoint_path)
    
    print(f'Input data shape: {input.shape}')
    
    prediction = model(input).detach().item()
    return prediction

