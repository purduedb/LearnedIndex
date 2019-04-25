import torch
import pandas as pd
import numpy as np
import csv
from torch.utils.data import Dataset
from torchvision import transforms, utils

class RtreeMappingDataset(Dataset):

    def __init__(self, csv_file, epsilon, transform=None):

        self.data_frame = pd.read_csv(csv_file, header=None, error_bad_lines=False,
                                      delimiter=' ', usecols=[0,1,2,3,4,5,6], skip_blank_lines=False)
        self.transform = transform
        self.epsilon = epsilon

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        inp = self.data_frame.iloc[idx, 0:2].as_matrix()
        idx = int(self.data_frame.iloc[idx, 6])

#        xrand = self.epsilon*np.random.rand( 49) + inp[0]
#        yrand = self.epsilon*np.random.rand(49) + inp[1]
#        inp = np.concatenate((inp, xrand, yrand))

        inp = torch.from_numpy(inp).float()
        if self.transform:
            sample = self.transform(sample)

        return (inp, idx)


