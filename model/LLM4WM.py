"""
   final-version
"""
import os
import numpy as np
import torch
import math
import torch.nn as nn
from model.modeling_gpt2 import GPT2Model
from model.modeling_llama import LlamaModel
from peft import MMOELoraConfig2, get_peft_model
from einops import rearrange
from peft import LoraConfig
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# 定义DFT码本生成函数
def dft_codebook(num_beams, num_antennas):
    D = np.zeros((num_antennas, num_beams), dtype=complex)
    for i in range(num_beams):
        D[:, i] = np.exp(-1j * 2 * np.pi * np.arange(num_antennas) * i / num_beams) / np.sqrt(num_antennas)
    return D


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Res_block_complex(nn.Module):
    def __init__(self, in_planes):
        super(Res_block_complex, self).__init__()

        self.linear1 = nn.Conv1d(in_planes, in_planes, 3, 1, 1)
        self.linear2 = nn.Conv1d(in_planes, in_planes, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        rs1 = self.relu(self.linear1(x))
        rs1 = self.linear2(rs1)
        rs = torch.add(x, rs1)
        return rs


def generate_res_block_complex(in_channel, num_res=1):
    layers = []
    for i in range(num_res):
        layers.append(Res_block_complex(in_planes=in_channel))
        # layers.append(nn.BatchNorm1d(in_channel))
    return nn.Sequential(*layers)


def generate_mlp_complex(dims, depth=1):
    layers = []
    for i in range(depth):
        layers.append(
            nn.Sequential(
                nn.Linear(dims, dims),
                nn.ReLU(),
                nn.Linear(dims, dims)
            )
        )
    return nn.Sequential(*layers)


class Muti_task_Adapter(nn.Module):

    def __init__(self, input_lens=16, input_dims=768, output_dims=768, n_adapter=3, task_id=1):
        super(Muti_task_Adapter, self).__init__()
        self.back_bone1 = generate_res_block_complex(input_lens, num_res=n_adapter)
        self.back_bone2 = generate_res_block_complex(input_lens, num_res=n_adapter)
        self.norm = nn.LayerNorm(output_dims)
        self.relu = nn.ReLU()
        self.dim_projection = nn.Sequential(
            nn.Linear(input_dims, output_dims),
        )

    def forward(self, x):
        x = self.dim_projection(x)
        h = self.back_bone1(x)
        out = self.back_bone2(self.relu(h))
        out = self.norm(out)
        return out


class Muti_task_output_projection(nn.Module):

    def __init__(self, input_dims, input_len, output_dims, output_len, type='mlp'):
        super(Muti_task_output_projection, self).__init__()
        self.relu = nn.ReLU()
        self.type = type
        if type == 'mlp':
            self.dim_in = input_dims * input_len
            self.dim_out = output_dims * output_len
            layers = []
            layers.append(nn.Linear(self.dim_in, self.dim_out))
            self.mlp = nn.Sequential(*layers)
        elif type == 'cnn':
            self.decoder_norm = nn.LayerNorm(input_dims)
            self.back_bone1 = generate_res_block_complex(input_len, num_res=2)
            self.back_bone2 = generate_res_block_complex(input_len, num_res=2)
            self.linear_projection_dim = nn.Sequential(

                nn.Linear(input_dims, output_dims)
            )
            self.linear_projection_len = nn.Sequential(

                nn.Linear(input_len, output_len)
            )

    def forward(self, x):
        if self.type == 'mlp':
            x = torch.flatten(x, 1)
            outs = self.mlp(x)
        elif self.type == 'cnn':
            h = self.back_bone1(x)
            h = self.back_bone2(self.relu(h))
            h = h + x
            h = self.decoder_norm(h)
            outs = self.linear_projection_dim(h)
            outs = self.linear_projection_len(outs.permute(0, 2, 1)).permute(0, 2, 1)

        return outs


class LLM4WM(nn.Module):

    def __init__(self, gpu_id=0, llm_name='gpt2', llm_layers=6, d_model=512, peft='moe',
                 lora_r=8, expert_num=8,
                 prev_len=16, pred_len=4, Nt1=8, Nt2=64, num_polit=16, K=64, num_beam=256,
                 train_stage=1,
                 is_llm_rand_inital=0, is_llm_frozen=0, is_llm_inference=1,
                 adapter_num=[0, 0],
                 dropout=0.1, is_sparse=0,
                 task_range=None, task_num=6):
        super(LLM4WM, self).__init__()
        assert len(task_range) == task_num
        self.device = torch.device('cuda:{}'.format(gpu_id))
        self.task_num = task_num
        self.adapter_num = adapter_num
        self.dropout = dropout
        # Task related parameters
        self.Nt1 = Nt1
        self.Nt2 = Nt2
        self.prev_len = prev_len
        self.pred_len = pred_len
        self.num_polit = num_polit
        self.num_data = K - num_polit
        self.K = K  # num of subcarriers in mmWave band
        self.d_model = d_model
        self.num_beam = num_beam  # num of DFT codebook
        # LLM related parameters
        self.llm_name = llm_name
        self.expert_num = expert_num
        self.peft = peft
        self.lora_r = lora_r
        self.is_moe = 0
        self.is_llm_rand_inital = is_llm_rand_inital
        self.is_llm_frozen = is_llm_frozen
        self.is_sparse = is_sparse
        self.is_llm_inference = is_llm_inference

        if train_stage == 1:
            self.is_llm_frozen = 1
        self.config = {
            'train_stage': train_stage,
            'task_num': self.task_num,
            'llm_name': self.llm_name,
            'is_llm_inference': self.is_llm_inference,
            'is_llm_rand_inital': self.is_llm_rand_inital,
            'is_llm_frozen': self.is_llm_frozen,
            'peft': self.peft,
            'is_sparse': self.is_sparse,
            'lora_r': self.lora_r,
            'expert_num': self.expert_num,
            'adapter_num': self.adapter_num
        }
        # Feature shape input to the Adapter
        self.Task_in_shape = {
            1: [self.prev_len, self.num_polit * self.Nt1 * 2],
            2: [self.prev_len, self.num_polit * self.Nt1 * 2],
            3: [self.prev_len, self.num_polit * self.Nt1 * 2],
            4: [self.num_polit, self.Nt2 * 2],
            5: [self.num_polit, self.Nt2 * 2],
            6: [self.num_polit, self.Nt2 * 2],
        }
        # Feature shape of each task's label
        self.Task_out_shape = {
            1: [self.prev_len, self.K * self.Nt1 * 2],
            2: [self.pred_len, self.num_polit * self.Nt1 * 2],
            3: [self.prev_len, self.num_polit * self.Nt1 * 2],
            4: [1, self.num_beam],
            5: [1, 1],
            6: [1, 1],
        }
        self.input_dim = K * 2

        # 1.Preprocess

        # 2.Muti-task-adapter

        # 3.LLM
        if self.llm_name == 'gpt2':
            self.llm = GPT2Model.from_pretrained('./pretrain/gpt2',
                                                 output_attentions=False, output_hidden_states=True)
            self.llm.h = self.llm.h[:llm_layers]
            self.hidden_dim_gpt = 768
            self.is_llama = 0

        elif self.llm_name == 'gpt2-medium':
            self.llm = GPT2Model.from_pretrained('./pretrain/gpt2-medium',
                                                 output_attentions=False, output_hidden_states=True)
            self.llm.h = self.llm.h[:llm_layers]
            self.hidden_dim_gpt = 1024
            self.is_llama = 0

        elif self.llm_name == 'gpt2-large':
            self.llm = GPT2Model.from_pretrained('./pretrain/gpt2-large',
                                                 output_attentions=False, output_hidden_states=True)
            self.llm.h = self.llm.h[:llm_layers]
            self.hidden_dim_gpt = 1280
            self.is_llama = 0

        elif self.llm_name == 'gpt2-xl':
            self.llm = GPT2Model.from_pretrained('./pretrain/gpt2-xl',
                                                 output_attentions=False, output_hidden_states=True)
            self.llm.h = self.llm.h[:llm_layers]
            self.hidden_dim_gpt = 1600
            self.is_llama = 0
        elif self.llm_name == 'llama':
            self.llm = LlamaModel.from_pretrained('./pretrain/Llama3.2-1B',
                                                  output_attentions=False, output_hidden_states=True)
            self.llm.layers = self.llm.layers[:llm_layers]
            self.hidden_dim_gpt = 2048
            self.is_llama = 1


        self.encoder_adapter_task = nn.ModuleList([
            Muti_task_Adapter(
                input_lens=self.Task_in_shape[i][0],
                input_dims=self.Task_in_shape[i][1],
                output_dims=self.hidden_dim_gpt,
                n_adapter=self.adapter_num[0],
                task_id=i)
            for i in task_range])

        # 4.Muti-task-adapter
        self.decoder_adapter_task = nn.ModuleList([
            Muti_task_Adapter(
                input_lens=self.Task_in_shape[i][0],
                input_dims=self.hidden_dim_gpt,
                output_dims=self.hidden_dim_gpt,
                n_adapter=self.adapter_num[1],
                task_id=i)
            for i in task_range])

        # 5.output-projection
        self.out_task = nn.ModuleList(self.Get_output(self.Task_in_shape[i], self.Task_out_shape[i], task_id=i)
                                      for i in task_range)

        # 6. set train parms
        self.stage = train_stage
        self.Set_train_parms(self.config)

    def Set_train_parms(self, config):
        print(f"Training Stage: {config['train_stage']}")
        print(f"Adapter Number: {config['adapter_num']}")
        print(f"LLM stage: | is_exist: {config['is_llm_inference']}"
              f" | is_frozen: {config['is_llm_frozen']}"
              f" | is_rand_init: {config['is_llm_rand_inital']}"
              f" | peft: {config['peft']}"
              f" | is_sparse: {config['is_sparse']}"
              f" | lora_r: {config['lora_r']}")
        if self.is_llama:
            target_module = ['up_proj', 'down_proj', 'gate_proj']
        else:
            target_module = ['c_fc', 'c_proj']
        if self.peft == 'moe':
            self.peft_config = MMOELoraConfig2(
                fan_in_fan_out=True,
                task_type="CAUSAL_LM",
                target_modules=target_module,
                inference_mode=False,
                r=self.lora_r, lora_alpha=self.lora_r * 2,
                lora_dropout=self.dropout,
                modules_to_save=[],
                task_num=self.task_num,
                task_embedding_dim=256,
                expert_num=self.expert_num,
                is_sparse=config['is_sparse']
            )
        else:
            self.peft_config = LoraConfig(
                r=self.lora_r,  # the dimension of the low-rank matrices
                lora_alpha=self.lora_r * 2,  # scaling factor for the weight matrices
                lora_dropout=self.dropout,  # dropout probability of the LoRA layers
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_module,
                fan_in_fan_out=True
            )
        if config['is_llm_rand_inital']:
            for name, param in self.llm.named_parameters():
                if param.requires_grad:
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.1)
        if config['is_llm_frozen'] == 1:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                param.requires_grad = False
        else:
            if config['train_stage'] == 1:
                for i, (name, param) in enumerate(self.llm.named_parameters()):
                    param.requires_grad = False
            elif config['train_stage'] == 2:
                if self.peft == 'moe':
                    self.is_moe = 1
                self.llm = get_peft_model(self.llm, self.peft_config)

    def Get_output(self, size_in, size_out, task_id):
        if task_id in [4, 5, 6]:
            return Muti_task_output_projection(input_dims=self.hidden_dim_gpt, output_dims=size_out[1],
                                               input_len=size_in[0], output_len=size_out[0],
                                               type='mlp')
        elif task_id in [1, 2, 3]:
            return Muti_task_output_projection(input_dims=self.hidden_dim_gpt, output_dims=size_out[1],
                                               input_len=size_in[0], output_len=size_out[0],
                                               type='cnn')

    def llm_forward(self, x, task_id=1):
        if self.is_moe and self.is_llama:
            outputs = self.llm(inputs_embeds=x, task_id=task_id).hidden_states[-1]
        elif self.is_moe and not self.is_llama:
            outputs = self.llm(inputs_embeds=x, task_id=task_id).last_hidden_state
        elif not self.is_moe and self.is_llama:
            outputs = self.llm(inputs_embeds=x).hidden_states[-1]
        else:
            outputs = self.llm(inputs_embeds=x).last_hidden_state
        return outputs

    def pre_process(self, x, task_id):
        if task_id in [1, 2, 3]:  # CE CP FP
            return rearrange(x, 'b t n k e -> b t (n k e)')
        elif task_id in [4, 5, 6]:  # BF DE PE
            F_transformer = torch.tensor(dft_codebook(self.Nt2, self.Nt1)).to(torch.complex128).to(self.device)
            x_complex = torch.complex(x[..., 0], x[..., 1]).to(torch.complex128)
            x_complex = x_complex.permute(0, 2, 1)
            out = x_complex @ F_transformer
            out = torch.cat([out.real, out.imag], dim=-1).to(torch.float)
            return out

    def post_process(self, x, task_id):
        if task_id == 1:  # CE
            return rearrange(x, 'a b (c d e) -> a b c d e', c=self.Nt1, e=2)
        elif task_id == 2:  # CP
            return rearrange(x, 'a b (c d e) -> a b c d e', c=self.Nt1, e=2)
        elif task_id == 3:  # PF
            return rearrange(x, 'a b (c d e) -> a b c d e', c=self.Nt1, e=2)
        elif task_id == 4:  # BF
            return x.squeeze()
        elif task_id == 5:  # DE
            return x
        elif task_id == 6:  # PE
            return x

    def forward(self, Y, task_range=None):
        output = {}
        assert len(task_range) == self.task_num
        time_llm = 0
        for task_id, i in zip(task_range, range(len(task_range))):
            Y_p = Y[f'T{task_id}i']

            token_in_1 = self.pre_process(Y_p, task_id)
            token_in_1 = self.encoder_adapter_task[i](token_in_1)
            start_time = time.time()
            if self.is_llm_inference:
                token_in_1 = self.llm_forward(token_in_1, task_id=i)
            time_llm += (time.time() - start_time) / len(task_range)

            task_decoder = self.decoder_adapter_task[i](token_in_1)

            task_out = self.post_process(self.out_task[i](task_decoder), task_id)

            output[f'Out{task_id}'] = task_out

        return output


if __name__ == '__main__':
    import torch
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}')
    task_range = [1, 2, 3, 4, 5, 6]
    inputs = torch.rand(2, 16, 8, 8, 2).to(device)
    inputs_2 = torch.rand(2, 8, 8, 2).to(device)
    # peft: ['moe', 'lora']
    model = LLM4WM(gpu_id=gpu_id, task_num=len(task_range), lora_r=8, expert_num=8,
                  K=64, num_polit=8, Nt1=8, Nt2=64,
                  peft='lora', train_stage=1, llm_name='gpt2', llm_layers=2,
                  is_llm_inference=1, is_llm_frozen=0, is_sparse=0, is_llm_rand_inital=0,
                  task_range=task_range, adapter_num=[2, 2]).to(device)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))
    in_Y = {
        'T1i': inputs,
        'T2i': inputs,
        'T3i': inputs,
        'T4i': inputs_2,
        'T5i': inputs_2,
        'T6i': inputs_2,
    }
    # Inference at Stage 1
    out = model(in_Y, task_range=task_range)
    for key, values in out.items():
        print(key, values.shape)
    # Inference at Stage 2
    model.Set_train_parms({**model.config, 'train_stage': 2})
    out = model(in_Y, task_range=task_range)
    for key, values in out.items():
        print(key, values.shape)

