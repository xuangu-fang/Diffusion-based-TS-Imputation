import numpy as np
import torch
from torch.optim import Adam
from kernels import KernelRBF
from model_GPTF import GPTF, GPTF_time
import random
import tqdm
import utils
import data_loader
import time

# baselines of GPTF based models with three settings: 
# 1. static(no time step), 
# 2. dicrete time, 
# 3. continus time (concat with embedding)

torch.manual_seed(2)

args = utils.parse_args_GPTF_based_model()
print(args.dataset, '  ',args.method)

if args.dataset == 'mvlens':
    file_base_name = '../data/processed/mvlens_small/mv_small_week_' #.npy'
elif args.dataset == 'ufo':
    file_base_name = '../data/processed/ufo/ufo_week_'
elif args.dataset == 'twitch':   
    file_base_name = '../data/processed/twitch/twitch_sub_hour_'
elif args.dataset == 'dblp':  
    file_base_name = '../data/processed/dblp/dblp_year_'
elif args.dataset == 'ctr':  
    file_base_name = '../data/processed/ctr/ctr_hour_'

result_dict = {}
rmse_test = []
MAE_test = []

start_time = time.time()

for fold in range(args.num_fold):
    file_name = file_base_name + str(fold) + '.npy'
    hyper_para_dict = data_loader.data_loader_GPTF_base(args,file_name)

    print('device is ', hyper_para_dict['device'])
    # device = torch.device('cpu')

    if args.method == 'GPTF-time':
        model = GPTF_time(hyper_para_dict['train_ind'], hyper_para_dict['train_y'],\
             hyper_para_dict['train_time'], hyper_para_dict['U'], args.m, args.batch_size,hyper_para_dict['device'])
        
        loss_test_rmse,loss_test_MAE = model.train(hyper_para_dict['test_ind'], hyper_para_dict['test_y'],\
             hyper_para_dict['test_time'], args.lr, args.epoch)

    else: 
        # static or discrte
        model = GPTF(hyper_para_dict['train_ind'], hyper_para_dict['train_y'],\
            hyper_para_dict['U'], args.m, args.batch_size,hyper_para_dict['device'])
        
        loss_test_rmse,loss_test_MAE = model.train(hyper_para_dict['test_ind'], hyper_para_dict['test_y'],\
            args.lr, args.epoch)

    rmse_test.append(loss_test_rmse.cpu().numpy().squeeze())
    MAE_test.append(loss_test_MAE.cpu().numpy().squeeze())


rmse_array = np.array(rmse_test)
MAE_array = np.array(MAE_test)

result_dict['time'] = time.time() - start_time
result_dict['rmse_avg'] = rmse_array.mean()
result_dict['rmse_std'] = rmse_array.std()
result_dict['MAE_avg'] = MAE_array.mean()
result_dict['MAE_std'] = MAE_array.std()

utils.make_log_GPTF(args,result_dict)