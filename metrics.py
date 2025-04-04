# =======================================================================================================================
# =======================================================================================================================
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange

# =======================================================================================================================
# =======================================================================================================================
class Acc_k(nn.Module):
    def __init__(self, ):
        super(Acc_k, self).__init__()

    def forward(self, label, input, topk=1):
        # label/input: B, 64
        B, _ = label.shape
        input[input > 1] = 1
        mut = torch.sum(label * input, dim=-1)
        mut[mut != topk] = 0
        mut[mut == topk] = 1
        loss = 1 - (mut.sum() / B)
        return loss

def dft_codebook(num_beams, num_antennas):
    D = np.zeros((num_antennas, num_beams), dtype=complex)
    for i in range(num_beams):
        D[:, i] = np.exp(-1j * 2 * np.pi * np.arange(num_antennas) * i / num_beams) / np.sqrt(num_antennas)
    return D

class SE_Loss_mmWave(nn.Module):
    def __init__(self, snr=10, Num_codebook=256, device=torch.device("cuda:0")):
        super().__init__()
        self.SNR = snr
        self.device = device
        self.codebook = torch.tensor(dft_codebook(Num_codebook, 64), dtype=torch.complex128)  # 64, num_codebook

    def forward(self, h, beam_id):
        B, Nt, K = h.shape
        h = h.permute(0, 2, 1)
        beam_vetor = self.codebook[:, np.array(beam_id.cpu())].permute(1, 0).view(B, 1, -1).repeat(1, K, 1).to(h.device)  # B * K * Nt

        # 1. prepare data
        H_true = rearrange(h.unsqueeze(-1), 'b k m n -> (b k) m n').to(torch.complex128)  # B, K, Nt, Nr
        S_real = torch.diag(torch.ones(1, 1)).unsqueeze(0).repeat([B*K, 1, 1])  # b,1,1
        S_imag = torch.zeros([B*K, 1, 1])
        S = torch.complex(S_real, S_imag).to(device=h.device).to(torch.complex128)
        matmul0 = torch.matmul(H_true, S)
        fro = torch.norm(matmul0, p='fro', dim=(1, 2))  # B*K,1
        noise_var = (torch.pow(fro, 2) / (Nt * 1)) * pow(10, (-self.SNR / 10))
        # get D and D0
        D = rearrange(beam_vetor, 'b k n -> (b k) n').unsqueeze(-2)  # B*K, Nr, Nt
        D = torch.div(D, torch.norm(D, p=2, dim=(1, 2), keepdim=True))
        D0 = torch.adjoint(H_true)
        D0 = torch.div(D0, torch.norm(D0, p=2, dim=(1, 2), keepdim=True))
        # 3. get SE and SE0
        matmul1 = torch.matmul(D, H_true)
        matmul2 = torch.matmul(D0, H_true)

        noise_var = noise_var.unsqueeze(1).unsqueeze(1)  # B,1,1
        SE = -torch.log2(torch.det(torch.div(torch.pow(torch.abs(matmul1), 2), noise_var) + S))  # B
        SE = torch.mean(SE.real)

        SE0 = -torch.log2(torch.det(torch.div(torch.pow(torch.abs(matmul2), 2), noise_var) + S))  # B
        SE0 = torch.mean(SE0.real)

        return SE, SE0



class SE_Loss(nn.Module):
    def __init__(self, snr=10, device=torch.device("cuda:0")):
        super().__init__()
        self.SNR = snr
        self.device = device

    def forward(self, h, h0):
        # input : h:  B, Nt, Nr (complex)      h0: B, Nt, Nr (complex)
        # 1. prepare data
        SNR = self.SNR
        B, Nt, Nr = h.shape
        H = h   # B * Nr * Nt
        H0 = h0.to(h.device).to(h.dtype)  # B * Nr * Nt
        if Nr != 1:
            S_real = torch.diag(torch.ones(Nr, 1).squeeze()).unsqueeze(0).repeat([B, 1, 1])  # b,2 * 2
        elif Nr == 1:
            S_real = torch.diag(torch.ones(Nr, 1)).unsqueeze(0).repeat([B, 1, 1])  # b,1,1
        S_imag = torch.zeros([B, Nr, Nr])
        S = torch.complex(S_real, S_imag).to(device=h.device).to(h.dtype)
        matmul0 = torch.matmul(H0, S)
        fro = torch.norm(matmul0, p='fro', dim=(1, 2))  # B,1
        noise_var = (torch.pow(fro, 2) / (Nt * Nr)) * pow(10, (-SNR / 10))
        # 2. get D and D0
        D = torch.adjoint(H)
        D = torch.div(D, torch.norm(D, p=2, dim=(1, 2), keepdim=True))
        D0 = torch.adjoint(H0)
        D0 = torch.div(D0, torch.norm(D0, p=2, dim=(1, 2), keepdim=True))
        # 3. get SE and SE0
        matmul1 = torch.matmul(D, H0)
        matmul2 = torch.matmul(D0, H0)

        noise_var = noise_var.unsqueeze(1).unsqueeze(1)  # B,1,1
        SE = -torch.log2(torch.det(torch.div(torch.pow(torch.abs(matmul1), 2), noise_var) + S))  # B
        SE = torch.mean(SE.real)

        SE0 = -torch.log2(torch.det(torch.div(torch.pow(torch.abs(matmul2), 2), noise_var) + S))  # B
        SE0 = torch.mean(SE0.real)

        return SE, SE0


# Example usage
if __name__ == "__main__":
    device = torch.device('cuda:0')
    f = SE_Loss_mmWave()
    x1 = torch.rand(1, 64, 64)
    x2 = torch.tensor([1])
    loss, loss2 = f(x1, x2)
    print(loss.item(), loss2.item())


def NMSE_cuda(x_hat, x):
    power = torch.sum(x ** 2)
    mse = torch.sum((x - x_hat) ** 2)
    nmse = mse / power
    return nmse



class NMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x_hat, x)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse
