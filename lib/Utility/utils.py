########################
# importing libraries
########################
# system libraries
import os
import shutil
import subprocess
import torch
import torch.utils.data
import pandas as pd
import numpy as np
import ast


def save_checkpoint(state, is_best, save_path=None, filename='checkpoint.pth.tar'):
    """
    saves the checkpoint for the model at the end of a certain epoch

    Parameters:
        state (dictionary): dictionary containing model state-dict, optimizer state-dict etc
        is_best (bool): if True then this is the best model seen as of now 
        save_path (string): path to save the models to
        filename (string): file-name to be used for saving the model
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path, 'model_best.pth.tar'))


class GPUMem:
    def __init__(self, is_gpu):
        """
        reads the total gpu memory that the program can access and computes the amount of gpu memory available 

        Parameters:
            is_gpu (bool): says if the computation device is cpu or gpu
        """
        self.is_gpu = is_gpu
        if self.is_gpu:
            self.total_mem = self._get_total_gpu_memory()

    def _get_total_gpu_memory(self):
        """
        gets the total gpu memory that the program can access

        Returns:
            total gpu memory (float) that the program can access
        """
        total_mem = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.total",
                                             "--format=csv,noheader,nounits"])

        return float(total_mem[0:-1])  # gets rid of "\n" and converts string to float

    def get_mem_util(self):
        """
        gets the amount of gpu memory currently being used

        Returns:
            mem_util (float): amount of gpu memory currently being used
        """
        if self.is_gpu:
            # Check for memory of GPU ID 0 as this usually is the one with the heaviest use
            free_mem = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free",
                                                "--format=csv,noheader,nounits"])
            free_mem = float(free_mem[0:-1])    # gets rid of "\n" and converts string to float
            mem_util = 1 - (free_mem / self.total_mem)
        else:
            mem_util = 0
        return mem_util


def get_best_architectures(csv_path=None, replay_dict=None, save_path='./runs/'):
    # set variables
    if csv_path is not None:
        df_replay_dict = pd.read_csv(csv_path, delimiter=',')
    else:
        df_replay_dict = replay_dict

    reward_col = 'reward'
    sorted_replay_dict = df_replay_dict.sort_values(by=[reward_col])
    # disregard elements with reward 0 - those have been not trained because the network is too big
    sorted_replay_dict = sorted_replay_dict[sorted_replay_dict[reward_col] > 0]
    sorted_replay_dict['acc_last_val'] = sorted_replay_dict['acc_val_all_epochs'].\
        apply(ast.literal_eval).\
        apply(lambda x: x[-1])
    sorted_replay_dict['acc_last_train'] = sorted_replay_dict['acc_train_all_epochs']. \
        apply(ast.literal_eval). \
        apply(lambda x: x[-1])

    sorted_replay_dict_acc = sorted_replay_dict[['epsilon', reward_col, 'acc_best_val', 'acc_last_val',
                                                 'acc_last_train', 'acc_test_last', 'net']]
    # remove duplicate nets
    sorted_replay_dict_acc = sorted_replay_dict_acc.drop_duplicates(subset='net', keep='last')

    num_rows = len(sorted_replay_dict_acc)
    num_to_train = 6
    indices = [i for i in range(num_to_train)]
    indices += [int(np.ceil(num_rows / 2.) - np.floor(num_to_train / 2.) + i) for i in range(num_to_train)]
    indices += [num_rows - 1 - i for i in range(num_to_train)][::-1]
    chosen_sorted_replay_dict_acc = sorted_replay_dict_acc.iloc[indices]
    chosen_sorted_replay_dict_acc.to_csv(os.path.join(save_path, 'chosen_to_train.csv'))

    return chosen_sorted_replay_dict_acc.index.tolist()
