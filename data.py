import torch.utils.data as data
import torch
import numpy as np
import hdf5storage
from numpy import random

def LoadBatch(H):
    # H: ...     [tensor complex]
    # out: ..., 2  [tensor real]
    size = list(H.shape)
    H_real = np.zeros(size + [2])
    H_real[..., 0] = H.real
    H_real[..., 1] = H.imag
    H_real = torch.tensor(H_real, dtype=torch.float32)
    return H_real


def Transform_TDD_FDD(H, Nt=4, Nr=4):
    # H: B,T,mul    [tensor real]
    # out:B',Nt,Nr  [tensor complex]
    H = H.reshape(-1, Nt, Nr, 2)
    H_real = H[..., 0]
    H_imag = H[..., 1]
    out = torch.complex(H_real, H_imag)
    return out


def noise(H, SNR):
    sigma = 10 ** (- SNR / 10)
    add_noise = np.sqrt(sigma / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    add_noise = add_noise * np.sqrt(np.mean(np.abs(H) ** 2))
    return H + add_noise

class Dataset(data.Dataset):
    def __init__(self, file_path, ir=1, SNR=25, is_few=0, SEED=42, is_show=1):
        super(Dataset, self).__init__()

        self.SNR = SNR
        self.ir = ir
        db = hdf5storage.loadmat(file_path)
        # Shuffle and Segmentation Dateset
        Batch_num = db['T1i'].shape[0]
        np.random.seed(SEED)
        idx = np.arange(Batch_num)
        np.random.shuffle(idx)

        if is_few:
            idx = idx[::10]

        self.T4i = db['T4i']
        if self.T4i.shape[-1] == 8:
            self.T4i = LoadBatch(noise(self.T4i, self.SNR))[idx, ...]
        else:
            self.T4i = LoadBatch(noise(self.T4i, self.SNR))[idx, ...][..., ::8, :]
        self.T4o = torch.tensor(db['T4o'], dtype=torch.float32)[idx, ...]
        self.T4mmW = LoadBatch(db['T4mmW'])[idx, ...]
        self.T4mmW = self.T4mmW / self.T4mmW.std()  # B, N, K
        self.T4o2 = torch.tensor(db['T4o2'])[idx, ...]  # B, N, K

        self.T5i = db['T5i']
        self.T5i = LoadBatch(noise(self.T5i, self.SNR))[idx, ...]
        self.T5o = torch.tensor(db['T5o'], dtype=torch.float32)[idx, ...]

        self.T6i = db['T6i']
        self.T6i = LoadBatch(noise(self.T6i, self.SNR))[idx, ...]
        self.T6o = torch.tensor(db['T6o'], dtype=torch.float32)[idx, ...]

        self.T2i = db['T2i']
        self.T2i = LoadBatch(noise(self.T2i, self.SNR))[idx, ...]
        self.T2o = LoadBatch(noise(db['T2o'], self.SNR))[idx, ...]  # B, T, N, K, 2

        self.T1i = db['T1i']
        self.T1i = LoadBatch(noise(self.T1i, self.SNR))[idx, ...]
        self.T1o = LoadBatch(noise(db['T1o'], self.SNR))[idx, ...]  # B, T, N, K

        self.T3i = db['T3i']
        self.T3i = LoadBatch(noise(self.T3i, self.SNR))[idx, ...]
        self.T3o = LoadBatch(noise(db['T3o'], self.SNR))[idx, ...]  # B, T, N, K

        if is_show:
            print('Dataset info: ')
            print(
                f'Task1 in shape: {self.T1i.shape}\t'
                f'Task1 out shape: {self.T1o.shape}\n'
                f'Task2 in shape: {self.T2i.shape}\t'
                f'Task2 out shape: {self.T2o.shape}\n'
                f'Task3 in shape: {self.T3i.shape}\t'
                f'Task3 out shape: {self.T3o.shape}\n'
                f'Task4 in shape: {self.T4i.shape}\t'
                f'Task4 out shape: {self.T4o.shape}\n'
                f'Task5 in shape: {self.T5i.shape}\t'
                f'Task5 out shape: {self.T5o.shape}\n'
                f'Task6 in shape: {self.T6i.shape}\t'
                f'Task6 out shape: {self.T6o.shape}\n'
                f'Task4 label shape: {self.T4o2.shape}\n'
            )

    def __getitem__(self, index):
        return {
            "T1i": self.T1i[index, ...].float(),
            "T1o": self.T1o[index, ...].float(),
            "T2i": self.T2i[index, ...].float(),
            "T2o": self.T2o[index, ...].float(),
            "T3i": self.T3i[index, ...].float(),
            "T3o": self.T3o[index, ...].float(),
            "T4i": self.T4i[index, ...].float(),
            "T4o": self.T4o[index, ...].float(),
            "T4mmW": self.T4mmW[index, ...].float(),
            "T4o2": self.T4o2[index, ...].long(),
            "T5i": self.T5i[index, ...].float(),
            "T5o": self.T5o[index, ...].float(),
            "T6i": self.T6i[index, ...].float(),
            "T6o": self.T6o[index, ...].float(),
        }

    def __len__(self):
        return self.T1i.shape[0]



def topk_to_one(tensor, k):
    topk_indices = torch.topk(tensor, k, dim=1).indices
    result = torch.zeros_like(tensor)
    result.scatter_(1, topk_indices, 1)
    return result

