import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset



class SIMU_Dataset(Dataset):
    def __init__(self, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0):
        
        np.random.seed(seed)  # seed for ground truth choice

        data_dict = np.load('./data/simu/simu_data_noise.npy',allow_pickle=True)


        x_raw = data_dict.item().get('time_step')
        y_raw = data_dict.item().get('data_all')
        observed_masks = data_dict.item().get('mask_train')

        # process to build train/test_x/y, and cooresponding index from the mask

        N_f,N = y_raw.shape 
        N_train = mask_train[0].sum()
        N_test = N-N_train

        path = (
            "./data/simu/simu_missing" + str(N_train/N) + "_seed" + str(seed) + ".pk"
        )

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            observed_values = y_raw # size: N_f * N (8*100)
            observed_masks = observed_masks.astype("float32")

            # gt_mask of simu data here is different from pm25/physio
            # as we don't have missing values,
            #  just set gt_masks = observed_masks = train data
            # also have to modify model.validation()

            gt_masks = observed_masks.astype("float32")



            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)
            self.gt_masks = np.array(self.gt_masks)



            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks], f
                )
        else:  # load datasetfile
                with open(path, "rb") as f:
                    self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                        f
                    )

        self.use_index_list = np.arange(len(self.observed_values))


        self.eval_length = N # ? to be confirmed later

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
    dataset = Physio_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    # start = (int)(nfold * 0.2 * len(dataset))
    # end = (int)((nfold + 1) * 0.2 * len(dataset))
    # test_index = indlist[start:end]
    # remain_index = np.delete(indlist, np.arange(start, end))

    # np.random.seed(seed)
    # np.random.shuffle(remain_index)
    # num_train = (int)(len(dataset) * 0.7)
    # train_index = remain_index[:num_train]
    # valid_index = remain_index[num_train:]

    dataset = Physio_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    
    valid_dataset = Physio_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    
    test_dataset = Physio_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    
    return train_loader, valid_loader, test_loader
