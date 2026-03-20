# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import re
import warnings
import math
from dataclasses import dataclass, field
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.pytorch_utils import Conv1D

from ..utils import (
    TRANSFORMERS_MODELS_TO_MMOELORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
)


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


@dataclass
class MMOELoraConfig2(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MMOELora`]
    """
    task_num: int = field(default=2, metadata={"help": "The number of tasks."})
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=4)
    is_sparse: int = field(default=1)

    def __post_init__(self):
        self.peft_type = PeftType.MMOELORA2


class MMOELoraModel2(LoraModel):
    """
    Create MMOELoRA (MMOE based LoRA) model from a pretrained transformers model.
    """

    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:  # get the lora config
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_mmoelora_config(config, model_config)  # load config
            self.peft_config[adapter_name] = config  # subsititue the original config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MMOELoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_num": lora_config.task_num,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
        }
        key_list = [key for key, _ in self.model.named_modules()]  # all module in raw model
        for key in key_list:
            # find the corresponding modules. target module has been split into list.
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                if isinstance(target, MMOELoraLayer2):
                    target.update_layer(
                        adapter_name,
                        lora_config.init_r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        raise NotImplementedError
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = MMOELoraLinear2(adapter_name, in_features, out_features,
                                                     bias=bias, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _prepare_mmoelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_MMOELORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_MMOELORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    # 0822 add
    def display_expert_weights(self, lora_config, task_id):
        weights_all = []
        # lora_config = self.peft_config[adapter_name]
        # print(lora_config)
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_num": lora_config.task_num,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
        }
        key_list = [key for key, _ in self.model.named_modules()]  # all module in raw model
        for key in key_list:
            # find the corresponding modules. target module has been split into list.
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                if isinstance(target, MMOELoraLinear2):

                    # print(parent, target, target_name)
                    task_id_tensor = torch.tensor([task_id], dtype=torch.int32).to(next(self.model.parameters()).device)

                    for adapter_name in target.lora_task_embedding.keys():
                        # Get the embedding for the task
                        task_embedding = target.lora_task_embedding[adapter_name](task_id_tensor)

                        # Calculate the expert weights using the gate network
                        expert_weight = target.lora_gate[adapter_name](task_embedding).squeeze(0).tolist()

                        weights_all.append(expert_weight)

                        # print(f"Adapter: {adapter_name}")
                        # print(f"Task ID: {task_id}")
                        # for i, weight in enumerate(expert_weight):
                        #     print(f"Expert {i + 1}: Weight = {weight:.4f}")
                        # print("\n")
        weight_mean = np.mean(np.array(weights_all), axis=0, keepdims=False)

        for i, weight in enumerate(weight_mean):
            print(f"Expert {i + 1}: Weight = {weight:.4f}")
        print("\n")


class MMOELoraLayer2(LoraLayer):

    def __init__(self, in_features: int, out_features: int, expert_num: int):

        super().__init__(in_features, out_features)
        self.expert_num = expert_num

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: MMOELinearA(self.in_features, r, self.expert_num)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: MMOELinearB(r, self.out_features, self.expert_num)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)


class MMOELoraLinear2(nn.Linear, MMOELoraLayer2):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MMOELoraLayer is the designed trainable Lora
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", True)
        self.task_num = kwargs.pop("task_num", True)
        self.te_dim = kwargs.pop("task_embedding_dim", True)
        self.is_sparse = kwargs.pop("is_sparse", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MMOELoraLayer2.__init__(self, in_features=in_features,
                                out_features=out_features,
                                expert_num=self.expert_num)

        # init the Gate network
        self.lora_task_embedding = nn.ModuleDict({})
        self.lora_gate = nn.ModuleDict({})
        self.lora_task_embedding.update(nn.ModuleDict({adapter_name: nn.Embedding(self.task_num + 1, self.te_dim)}))
        self.lora_gate.update(nn.ModuleDict({adapter_name: Gate(self.te_dim, self.expert_num)}))

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self, task_id):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            expert_weight = self.lora_gate[self.active_adapter](self.lora_task_embedding[self.active_adapter](task_id))
            for i in range(self.expert_num):
                lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
                lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
                self.weight.data += (
                        transpose(
                            lora_B_weights @ lora_A_weights,
                            self.fan_in_fan_out,
                        )
                        * self.scaling[self.active_adapter]
                        * expert_weight[..., i]
                )
            self.merged = True

    def unmerge(self, task_id):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            expert_weight = self.lora_gate[self.active_adapter](self.lora_task_embedding[self.active_adapter](task_id))
            for i in range(self.expert_num):
                lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
                lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
                self.weight.data -= (
                        transpose(
                            lora_B_weights @ lora_A_weights,
                            self.fan_in_fan_out,
                        )
                        * self.scaling[self.active_adapter]
                        * expert_weight[..., i]
                )
            self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):
        # print(kwargs)
        ids = kwargs["task_id"]
        previous_dtype = x.dtype
        previous_device = x.device
        task_id = torch.tensor([ids], dtype=torch.int32).to(previous_device)

        if self.active_adapter not in self.lora_A.keys():  # No adapter, directly use linear
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:  # No adapter
            if self.r[self.active_adapter] > 0 and self.merged:  # merge the adapter to linear
                self.unmerge(task_id)
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:  # general lora process
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)

            expert_weight = self.lora_gate[self.active_adapter](
                self.lora_task_embedding[self.active_adapter](task_id))  # 1, exper_num
            if not self.is_sparse:
                # dense MoE
                for i in range(self.expert_num):
                    result += (  # lora process
                            self.lora_B[self.active_adapter].loraB[i](
                                self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                            )
                            * self.scaling[self.active_adapter]
                            * expert_weight[..., i].unsqueeze(-1).unsqueeze(0)
                    )

            # sparse MoE
            else:
                select_id = torch.argmax(expert_weight, dim=1)  # 1,
                result += (  # lora process
                        self.lora_B[self.active_adapter].loraB[select_id](
                            self.lora_A[self.active_adapter].loraA[select_id](
                                self.lora_dropout[self.active_adapter](x)),
                        )
                        * self.scaling[self.active_adapter]
                        * expert_weight[..., select_id].unsqueeze(0)
                )

        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype).to(previous_device)

        return result


class MMOELinearA(nn.Module):
    '''MMOE based LoRA block'''

    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])

        assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
        self.r = self.out_features // self.expert_num

        for _ in range(self.expert_num):
            self.loraA.append(Expert(self.in_features, self.r))

    def forward(self, x):
        '''input x is a vector, return output is a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraA[i](x))

        return outputs


class MMOELinearB(nn.Module):
    '''MMOE based LoRA block'''

    def __init__(self, in_features, out_features, expert_num) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])

        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num

        for _ in range(self.expert_num):
            self.loraB.append(Expert(self.r, self.out_features))

    def forward(self, x):
        '''input x is a list, return output is also a list'''
        outputs = []
        for i in range(self.expert_num):
            outputs.append(self.loraB[i](x[i]))

        return outputs


class Expert(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight

    def forward(self, x):
        # LoRA A or B block
        y = self.mlp(x)

        return y


class Gate(nn.Module):

    def __init__(self, input_size, expert_num):
        super().__init__()
        # 使用embedding来代替线性层
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)  # 第0维为batch size

    def forward(self, x):
        y = self.GateL(x)
        y = self.act(y)

        return y








