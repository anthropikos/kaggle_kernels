# Anthony Lee 2025-01-05

from typing import Union
from pathlib import Path
import torch
from torch import nn

def checkpoint_save(epoch:int, 
                    save_path:Union[str, Path],
                    model:nn.Module, 
                    monet_gen_optim:torch.optim.Optimizer,
                    photo_gen_optim:torch.optim.Optimizer,
                    monet_dis_optim:torch.optim.Optimizer,
                    photo_dis_optim:torch.optim.Optimizer,
                    loss_tracker:dict,
                    ):
    filename = f"torch_epoch_{epoch}.checkpoint"

    save_path = (Path(save_path)/Path(filename)).resolve()

    dict_to_serialize = {
        "epoch": epoch, 
        "model_state_dict": model.state_dict(),
        "monet_gen_optim":monet_gen_optim.state_dict(),
        "photo_gen_optim":photo_gen_optim.state_dict(),
        "monet_dis_optim":monet_dis_optim.state_dict(),
        "photo_dis_optim":photo_dis_optim.state_dict(),
        "loss_tracker": loss_tracker,
    }
    torch.save(dict_to_serialize, save_path)
    return dict_to_serialize
