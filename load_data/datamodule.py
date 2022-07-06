import pickle
import numpy as np
import torch
import torch.utils.data as Data
from load_data.data_params import get_dataInfo
from tqdm import tqdm

dataname, file_path, unit_time, _, _, m = get_dataInfo()

class CascadeData(Data.Dataset):
    def __init__(self, file_path, data_name, observation, file_type="train"):
        super(CascadeData, self).__init__()
        with open(file_path + data_name + "_{}/{}.pkl".format(observation, file_type), 'rb') as f:
            self.dataset = pickle.load(f)
            self.ln = len(self.dataset["labels"])

    def __getitem__(self, item):
        return self.dataset["cascade_id"][item], self.dataset["cascade_src"][item], self.dataset["temporal_src"][item], \
               self.dataset["spl_matrix"][item], self.dataset["td_matrix"][item], self.dataset["lca_matrix"][item], \
               self.dataset["lpe_matrix"][item], self.dataset["labels"][item]

    def __len__(self):
        return self.ln


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    cascade_id, cascade_src, temporal_src, spl_matrix, td_matrix, lca_matrix, lpe_matrix, labels = zip(*batch)
    # obtain the padding length
    real_len = [len(s) for s in cascade_src]
    max_len = max(real_len)
    # generate shortest path length matrix, time interval difference matrix and lca matrix
    pad_src, pad_time = [], []
    pad_spl_lst, pad_td_lst, pad_lca_lst, pad_lpe_list = [], [], [], []

    for item in range(len(labels)):
        l = real_len[item]

        temp_cas_src = [0] * max_len
        temp_cas_src[:l] = cascade_src[item]
        pad_src.append(temp_cas_src)

        temp_temp_src = [0] * max_len
        temp_temp_src[:l] = temporal_src[item]
        pad_time.append(temp_temp_src)

        pad_lca_matrix = np.zeros((max_len, max_len))
        pad_lca_matrix[:l,:l] = lca_matrix[item]
        pad_lca_lst.append(pad_lca_matrix)

        pad_spl_matrix = np.zeros((max_len, max_len))
        pad_spl_matrix[:l,:l] = spl_matrix[item]
        pad_spl_lst.append(pad_spl_matrix)

        pad_td_matrix = np.zeros((max_len, max_len))
        pad_td_matrix[:l,:l] = td_matrix[item]
        pad_td_lst.append(pad_td_matrix)

        pad_lpe_matrix = np.zeros((max_len, m))
        pad_lpe_matrix[:l,:m] = lpe_matrix[item]
        pad_lpe_list.append(pad_lpe_matrix)
    # 生成lca idx
    bz, n = len(labels), max_len
    pad_src = torch.LongTensor(pad_src)
    pad_lca_lst = torch.LongTensor(pad_lca_lst)
    aux_source = pad_src.unsqueeze(-1).repeat(1, 1, n)
    aux_target = pad_src.repeat(1, 1, n).view(bz, n, n)
    pad_lca_lst = torch.cat((aux_source, pad_lca_lst, aux_target), -1)    # (b, n, 3n)

    return pad_src, torch.FloatTensor(pad_time), torch.FloatTensor(pad_lpe_list), \
           torch.LongTensor(pad_spl_lst), torch.FloatTensor(pad_td_lst), pad_lca_lst, \
           torch.FloatTensor(labels), torch.LongTensor(real_len)


def creat_dataloader(file_type, data_name, observation, batch_size, shuffle=True):
    dataset = CascadeData(file_path, data_name, observation, file_type)
    loader = Data.DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader

def obtain_N(data_name, observation):
    with open(file_path + data_name + "_{}/u2idx.pkl".format(observation), 'rb') as f:
        u2idx_dict = pickle.load(f)
    return len(u2idx_dict)

