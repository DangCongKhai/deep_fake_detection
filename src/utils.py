import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

# NOTE - Set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

# NOTE - FFPPTorchDataset(Dataset)
# class FFPPTorchDataset(Dataset)