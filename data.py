import torch.utils.data as data
import torch
import numpy as np
import hdf5storage
from numpy import random
from metrics import Acc_k, dft_codebook


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


# SNR 5, 15, 20
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

    def Loc_quantization(self, x, Fine_grained=100):
        assert len(x.shape) == 2
        min_val = x.min(dim=0, keepdim=True)[0]
        max_val = x.max(dim=0, keepdim=True)[0]
        x_norm = (x - min_val) / (max_val - min_val)
        output = torch.ceil(x_norm * Fine_grained) / Fine_grained
        return output


def topk_to_one(tensor, k):
    # 获取每行最大的 k 个值的索引
    topk_indices = torch.topk(tensor, k, dim=1).indices
    # 创建一个全零张量
    result = torch.zeros_like(tensor)
    # 在 top-k 的位置上设置为 1
    result.scatter_(1, topk_indices, 1)
    return result


def topk_to_one_trainable(tensor, k):
    # 获取每行最大的 k 个值的索引
    topk_indices = torch.topk(tensor, k, dim=1).indices
    # 创建一个全零张量
    result = torch.zeros_like(tensor) + 1e-6
    # 在 top-k 的位置上设置为 1
    result.scatter_(1, topk_indices, 1)
    output = tensor * result  # + (1e-6) * (1 - result)
    # output = result
    return output


def get_Relevance_mmW_sub6(dataset):
    lens = len(dataset)
    loss_func = Acc_k()
    acc_1 = []
    acc_4 = []
    acc_16 = []
    for iter in range(lens):
        mmW = dataset[iter]["mmWave"].squeeze()
        mmW_com = torch.complex(mmW[..., 0], mmW[..., 1])  # 64, 64
        mmW_com = torch.abs(torch.fft.fft(mmW_com, dim=0))  # 64, 64
        mmW_com = torch.sum(mmW_com, dim=1, keepdim=False)
        mmW_com = mmW_com / torch.max(mmW_com)

        beam_label = topk_to_one(mmW_com.unsqueeze(0), k=1)

        sub6 = dataset[iter]["H_sub6"][15, ...].squeeze()
        sub6_com = torch.complex(sub6[..., 0], sub6[..., 1])  # 64, 64
        zeros = torch.zeros([mmW_com.shape[0] - sub6_com.shape[0], sub6_com.shape[1]])
        # print(sub6_com.shape, zeros.shape)
        sub6_com = torch.cat([sub6_com, zeros], dim=0)
        sub6_com = torch.abs(torch.fft.fft(sub6_com, dim=0))  # 64, 64
        sub6_com = torch.sum(sub6_com, dim=1, keepdim=False)
        sub6_com = sub6_com / torch.max(sub6_com)

        acc_1.append(loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=1)))
        acc_4.append(loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=4)))
        acc_16.append(loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=16)))

    print(np.nanmean(np.array(acc_1)))  # 0.5888877
    print(np.nanmean(np.array(acc_4)))  # 0.5041072
    print(np.nanmean(np.array(acc_16)))  # 0.3240955


def get_Relevance_beamlabel_sub6(dataset):
    lens = len(dataset)
    loss_func = Acc_k()
    acc_1 = []
    acc_4 = []
    acc_16 = []
    for iter in range(lens):
        # mmW = dataset[iter]["mmWave"].squeeze(-2)
        # mmW_com = torch.complex(mmW[..., 0], mmW[..., 1])  # 64, 64
        # mmW_com = torch.abs(torch.fft.fft(mmW_com, dim=1))  # 64, 64
        # mmW_com = torch.sum(mmW_com, dim=0, keepdim=False)
        # mmW_com = mmW_com / torch.max(mmW_com)
        #
        # beam_label = topk_to_one(mmW_com.unsqueeze(0), k=1)

        beam_label = dataset[iter]["T2o2"].unsqueeze(0)
        beam_label = topk_to_one(beam_label, k=1)

        sub6 = dataset[iter]["sub6"].squeeze().permute(1, 0, 2)  # 64, 16, 2
        # print(sub6.shape)
        sub6_com = torch.complex(sub6[..., 0], sub6[..., 1]).to(torch.complex128)  # 64, 16
        F_transformer2 = torch.tensor(dft_codebook(64, sub6_com.shape[1])).to(torch.complex128)
        Y_p_angle2 = torch.sum(torch.abs(sub6_com @ F_transformer2).to(torch.float), dim=0,
                               keepdim=False)  # B, n_p, Nt2

        sub6_com = Y_p_angle2 / torch.max(Y_p_angle2)

        acc1 = loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=1))
        acc4 = loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=4))
        acc16 = loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=16))

        # print(acc1, acc4, acc16)

        acc_1.append(loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=1)))
        acc_4.append(loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=4)))
        acc_16.append(loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=16)))

    print(np.nanmean(np.array(acc_1)))  # 0.9786647
    print(np.nanmean(np.array(acc_4)))  # 0.6398847
    print(np.nanmean(np.array(acc_16)))  # 0.3844287


def get_Relevance_beamlabel_sub6_2(dataset):
    lens = len(dataset)
    loss_func = Acc_k()
    acc_1 = []
    acc_4 = []
    acc_16 = []
    for iter in range(lens):
        # mmW = dataset[iter]["mmWave"].squeeze(-2)
        # mmW_com = torch.complex(mmW[..., 0], mmW[..., 1])  # 64, 64
        # mmW_com = torch.abs(torch.fft.fft(mmW_com, dim=1))  # 64, 64
        # mmW_com = torch.sum(mmW_com, dim=0, keepdim=False)
        # mmW_com = mmW_com / torch.max(mmW_com)
        #
        # beam_label = topk_to_one(mmW_com.unsqueeze(0), k=1)

        beam_label = dataset[iter]["T2o2"].unsqueeze(0)
        beam_label = topk_to_one(beam_label, k=1)

        sub6 = dataset[iter]["mmW"].squeeze().permute(1, 0, 2)  # 64, 16, 2
        # print(sub6.shape)
        sub6_com = torch.complex(sub6[..., 0], sub6[..., 1]).to(torch.complex128)  # 64, 16
        F_transformer2 = torch.tensor(dft_codebook(64, sub6_com.shape[1])).to(torch.complex128)
        Y_p_angle2 = torch.sum(torch.abs(sub6_com @ F_transformer2).to(torch.float), dim=0,
                               keepdim=False)  # B, n_p, Nt2

        sub6_com = Y_p_angle2 / torch.max(Y_p_angle2)

        acc1 = loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=1))
        acc4 = loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=4))
        acc16 = loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=16))

        # print(acc1, acc4, acc16)

        acc_1.append(loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=1)))
        acc_4.append(loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=4)))
        acc_16.append(loss_func(beam_label, topk_to_one(sub6_com.unsqueeze(0), k=16)))

    print(np.nanmean(np.array(acc_1)))  # 0.9786647
    print(np.nanmean(np.array(acc_4)))  # 0.6398847
    print(np.nanmean(np.array(acc_16)))  # 0.3844287


def test_codebook(dataset):
    lens = len(dataset)
    loss_func = Acc_k()
    acc_1 = []
    acc_2 = []

    for iter in range(lens):
        beam_label = dataset[iter]["T4o"].unsqueeze(0)

        sub6 = dataset[iter]["T4i"].squeeze().permute(1, 0, 2)  # 8(K), 8(N), 2
        # print(sub6.shape)
        sub6_com = torch.complex(sub6[..., 0], sub6[..., 1]).to(torch.complex128)  # 64, 16
        F_transformer2 = torch.tensor(dft_codebook(256, sub6_com.shape[1])).to(torch.complex128)
        Y_p_angle2 = torch.sum(torch.abs(sub6_com @ F_transformer2).to(torch.float), dim=0,
                               keepdim=False)  # B, n_p, Nt2

        sub6_com = Y_p_angle2 / torch.max(Y_p_angle2)

        acc1 = loss_func(topk_to_one(beam_label, k=1), topk_to_one(sub6_com.unsqueeze(0), k=1), topk=1)
        acc2 = loss_func(topk_to_one(beam_label, k=2), topk_to_one(sub6_com.unsqueeze(0), k=2), topk=2)
        ids = torch.argmax(beam_label, dim=1)
        print(ids)
        acc_1.append(acc1)
        acc_2.append(acc2)

    print(np.nanmean(np.array(acc_1)))  # 0.9786647
    print(np.nanmean(np.array(acc_2)))  # 0.6398847


if __name__ == '__main__':
    batch_size = 32

    path1 = '/data1/PCNI1_data/MTLLM_v2/MTLLM_1106/H_train/norm/Dataset_uma_0107.mat'
    # 0.552
    # 0.6625
    data_set = Dataset(path1, SNR=10, is_train=0, test_per=0.1)
    # 0.631
    # 0.585
    test_codebook(data_set)
    # for key, value in data_set[0].items():
    #     print(key, value.shape)
    # print(data_set[0]["T1o"])
