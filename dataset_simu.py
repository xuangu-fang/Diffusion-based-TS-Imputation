import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset



class SIMU_Dataset(Dataset):
    def __init__(self, eval_length=50, use_index_list=None, missing_ratio=0.0, seed=0):
        
        np.random.seed(seed)  # seed for ground truth choice

        data_dict = np.load('./data/simu/simu_data_noise_new.npy',allow_pickle=True)

        '''
        'data_all':noise_data,# time series
        'data_shape':(N,K,L),
        'time_step':T_array,# Time step
        'mask_obs':mask_obs, 
        'mask_gt':mask_gt   
        '''

        
        self.observed_values= data_dict.item().get('data_all')
        self.observed_masks = data_dict.item().get('mask_obs')
        self.gt_masks = data_dict.item().get('mask_gt')

        N,L,K = self.observed_values.shape

        self.use_index_list = np.arange(N)


        self.eval_length = L 

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),# ? use the raw values?
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1):

    # only to obtain total length of dataset
    dataset = SIMU_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    dataset = SIMU_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    
    valid_dataset = SIMU_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    
    test_dataset = SIMU_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader
