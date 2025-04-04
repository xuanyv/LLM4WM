import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import Dataset, topk_to_one
from metrics import Acc_k, NMSELoss
from model.LLM4WM import LLM4WM


def Compute_loss(output, label, is_train=1):
    loss = [0, 0, 0, 0, 0, 0]
    for i in task_range:
        if i == 1:
            if is_train:
                loss[i - 1] = criterion_mse(output[f'Out{i}'], label['T1o'])
            else:
                loss[i - 1] = criterion_nmse(output[f'Out{i}'], label['T1o'])
        elif i == 2:
            if is_train:
                loss[i - 1] = criterion_mse(output[f'Out{i}'], label['T2o'])
            else:
                loss[i - 1] = criterion_nmse(output[f'Out{i}'], label['T2o'])
        elif i == 3:
            if is_train:
                loss[i - 1] = criterion_mse(output[f'Out{i}'], label['T3o'])
            else:
                loss[i - 1] = criterion_nmse(output[f'Out{i}'], label['T3o'])
        elif i == 4:
            if is_train:
                loss[i - 1] = criterion_CrossEntropyLoss(output[f'Out{i}'], label['T4o2'])
            else:
                mat1 = torch.zeros_like(output[f'Out{i}'])
                mat1.scatter_(1, label['T4o2'].unsqueeze(1), 1)
                loss[i - 1] = criterion_Acc(mat1, topk_to_one(output[f'Out{i}'], k=1))
        elif i == 5:
            if is_train:
                loss[i - 1] = criterion_mse(output[f'Out{i}'], label['T5o'])
            else:
                loss[i - 1] = criterion_mae(output[f'Out{i}'], label['T5o'])
        elif i == 6:
            if is_train:
                loss[i - 1] = criterion_mse(output[f'Out{i}'], label['T6o'])
            else:
                loss[i - 1] = criterion_nmse(output[f'Out{i}'], label['T6o'])

    return loss


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################
def inference(test_data_loader):
    print('Start Testing...')
    # ===============  Test  =============== #
    model.eval()
    test_loss = []
    with torch.no_grad():
        for iteration, batch in enumerate(test_data_loader, 1):
            for key, value in batch.items():
                batch[key] = batch[key].to(device)
            outputs = model(batch, task_range=task_range)
            loss_batch = Compute_loss(outputs, batch, is_train=0)
            for i in task_range:
                loss_batch[i - 1] = loss_batch[i - 1].item()
            test_loss.append(loss_batch)
        print(f'Test Results:', end=' ')
        loss_mean = np.nanmean(np.array(test_loss), axis=0)
        for loss, i in zip(loss_mean, range(len(loss_mean))):
            print(f'Task{i + 1} Loss: {loss}', end=' ')
        print()

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":

    Configs = {
        'Standard_gpr2_r(8)_ne(8)': {'tasks': [1, 2, 3, 4, 5, 6], 'lora_r': 8, 'expert_num': 8, 'adapter_num': [2, 2],
                        'is_llm_frozen': 0, 'is_llm_inference': 1, 'is_llm_rand_inital': 0,
                        'is_sparse': 0, 'llm_name': 'gpt2', 'peft': 'moe', 'snr': 10,
                        'pre_trained': './Weights/LLM4WM_standard.pth'},
    }
    for mode, config in Configs.items():
        batch_size = 256
        gpu_id = 0
        device = torch.device(f'cuda:{gpu_id}')

        task_range = config['tasks']
        expert_num = config['expert_num']
        lora_r = config['lora_r']
        adapter_num = config['adapter_num']

        is_sparse = config['is_sparse']
        is_llm_frozen = config['is_llm_frozen']
        is_llm_inference = config['is_llm_inference']
        is_llm_rand_inital = config['is_llm_rand_inital']

        is_few = 0
        SNR = config['snr']

        peft_type = config['peft']
        save_root = './Weights/'
        model_name = f'LLM4MT_mode_{peft_type}_lora_r{lora_r * expert_num}_na{adapter_num}_0110_{mode}_' \
                     f'{is_sparse}_{is_llm_frozen}_{is_llm_inference}_{is_llm_rand_inital}' + f'_task{task_range}.pth'
        # Init Model
        model = LLM4WM(gpu_id=gpu_id, train_stage=2,
                      adapter_num=adapter_num,
                      llm_layers=6, peft=peft_type, llm_name=config['llm_name'],
                      task_num=len(task_range), is_sparse=is_sparse,
                      is_llm_frozen=is_llm_frozen, is_llm_inference=is_llm_inference,
                      is_llm_rand_inital=is_llm_rand_inital,
                      expert_num=expert_num, lora_r=lora_r * expert_num, task_range=task_range).to(device)
        # Load Test Dataset
        path1 = './dataset/data_test.mat'
        test_set = Dataset(path1, is_few=is_few, SNR=SNR)  # creat data for test
        test_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
        # Load Pre-Trained Weights
        if config['pre_trained'] is not None:
            print(f"Loading pre-trained model from {config['pre_trained']}")
            model = torch.load(config['pre_trained'], map_location=device, weights_only=False)
            model.device = device
            setattr(model, 'is_llm_inference', 1)

        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.5fM" % (total / 1e6))
        total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

        criterion_mse = nn.MSELoss().to(device)
        criterion_mae = nn.L1Loss().to(device)
        criterion_nmse = NMSELoss().to(device)
        criterion_CrossEntropyLoss = nn.CrossEntropyLoss().to(device)
        criterion_Acc = Acc_k().to(device)

        inference(test_data_loader)
