##########################################################################
# Copyright 2022 Jianping Cai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################
# DPCREngine succeeds PrivacyEngine.
# Just replace "PrivacyEngine" in your Opacus code with "DPCREngine",
# you will complete your private learning with our Opacus-DPCR.
##########################################################################

import dpcrpy as dcr
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizers import DPOptimizer
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Callable, List, Tuple, Union
from dpcrpy.framework.dpcrMech import DpcrMech

from opacus import PrivacyEngine
from opacus_dpcr.dpcrOptimizer import DPCROptimizer


class DPCREngine(PrivacyEngine):

    def __init__(self, *, accountant: str = "rdp", secure_mode: bool = False,
                 dpcrGenerator: Callable[[int], DpcrMech] = lambda x: dcr.SimpleMech(1)):
        PrivacyEngine.__init__(self, accountant=accountant, secure_mode=secure_mode)
        self.dpcrGenerator = dpcrGenerator;

    def make_private(
            self,
            *,
            module: nn.Module,
            optimizer: optim.Optimizer,
            data_loader: DataLoader,
            noise_multiplier: float,
            max_grad_norm: Union[float, List[float]],
            batch_first: bool = True,
            loss_reduction: str = "mean",
            poisson_sampling: bool = True,
            clipping: str = "flat",
            noise_generator=None,
    ) -> Tuple[GradSampleModule, DPOptimizer, DataLoader]:
        if optimizer.state_dict()['param_groups'][0]['momentum'] > 0:
            mom = optimizer.state_dict()['param_groups'][0]['momentum'];
            raise RuntimeError(f"momentum = {mom} > 0. The gradient descent with momentum does not support currently.")
        module, optimizer, data_loader = PrivacyEngine.make_private(self, module=module, optimizer=optimizer,
                                                                    data_loader=data_loader,
                                                                    noise_multiplier=noise_multiplier,
                                                                    max_grad_norm=max_grad_norm,
                                                                    batch_first=batch_first,
                                                                    loss_reduction=loss_reduction,
                                                                    poisson_sampling=poisson_sampling,
                                                                    clipping=clipping,
                                                                    noise_generator=noise_generator)
        #! PrivacyEngine.make_private 使用了错误的sample_rate，见privacy_engine.py的372-373行以及391-393行
        optimizer.attach_step_hook(
            self.accountant.get_optimizer_hook_fn(sample_rate=data_loader.sample_rate)
        )
        ###########################
        optimizer = DPCROptimizer(optimizer, self.dpcrGenerator)
        return module, optimizer, data_loader

    def __str__(self):
        res='采用了DPCREngine实现DP-SGD';
        return res;