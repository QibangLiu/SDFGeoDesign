# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import NOT_SS
from models import configs
from torch.utils.data import DataLoader
from skimage import measure
from tqdm import tqdm
import json
# from torchinfo import summary
import time
from models.inverse_diffusion_from_sdf import LoadDiffusionInverseModel
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
fwd_config = configs.NOTSS_configs()
fwd_filebase = fwd_config["filebase"]
fwd_args = fwd_config["model_args"]
not_ss = NOT_SS.LoadNOTModel(fwd_filebase, fwd_args)

# %%
# %%
sdf = torch.randn(10, 1, 120, 120).to(device)
xy = torch.randn(51, 1).to(device)
o = not_ss(xy, sdf)

# %%
inv_config = configs.INV_configs()
inv_filebase = inv_config["filebase"]
inv_args = inv_config["model_args"]
fwd_config = configs.NOTSS_configs()
fwd_filebase = fwd_config["filebase"]
fwd_args = fwd_config["model_args"]

inv_Unet, gaussian_diffusion = LoadDiffusionInverseModel(
    inv_filebase, inv_args)

batch_size = 10
x_t = torch.randn(10, 1, 120, 120).to(device)
t = torch.randn(10).to(device)
c = torch.randn(10, 51).to(device)
# %%
out = inv_Unet(x_t, t, c, torch.zeros(batch_size).int().to(device))
