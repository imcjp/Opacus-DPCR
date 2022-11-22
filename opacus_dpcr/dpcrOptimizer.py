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
# Optimizer for DPCR learning, generated through DPCREngine.
##########################################################################
from opacus.optimizers import DPOptimizer
from torch.optim import Optimizer
from typing import Callable, Optional
import torch
import dpcrpy as dcr
import numpy as np
from opacus_dpcr.tensorNoiMech import TensorGaussNoiMech
from dpcrpy.framework.dpcrMech import DpcrMech


class DPCROptimizer(Optimizer):
    def __init__(self, optimizer: DPOptimizer, dcrMechGen: Callable[[int], DpcrMech]):
        self.optimizer = optimizer
        self.dpcrs = []
        #####################################
        rou = 1.0
        accumulated_iterations = 1
        if self.optimizer.loss_reduction == "mean":
            rou /= self.optimizer.expected_batch_size * accumulated_iterations
        self.gSigma = self.optimizer.noise_multiplier * self.optimizer.max_grad_norm * rou
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.gSigma *= lr
        #####################################
        self._shareId = 0;
        self.noiMeches = []
        genBlk = lambda x: dcrMechGen(x).setNoiMech(self.noiMeches[self._shareId])
        for param in self.optimizer.params:
            sz = np.array(param.size());
            noiMech = TensorGaussNoiMech(sigma0=self.gSigma, tsSize=sz, device=param.device)
            self.noiMeches.append(noiMech)
            dpcr = dcr.dpCrFw(genBlk)
            dpcr.initRelease(param)
            self.dpcrs.append(dpcr)
            self._shareId += 1;

    def pre_step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        if self.optimizer.params[0].grad_sample is not None:
            self.optimizer.clip_and_accumulate()
            if self.optimizer._check_skip_next_step():
                self.optimizer._is_last_step_skipped = True
                return False
            for p in self.optimizer.params:
                p.grad = p.summed_grad.view_as(p.grad)
            self.optimizer.scale_grad()
            if self.optimizer.step_hook:
                self.optimizer.step_hook(self.optimizer)
        else:
            self.accumulated_iterations=1
            self.noise_multiplier=self.optimizer.noise_multiplier
            if self.optimizer.step_hook:
                self.optimizer.step_hook(self)
        self.optimizer._is_last_step_skipped = False
        return True

    def step(self, incMode=False, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()
        if self.pre_step():
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            with torch.no_grad():
                self._shareId = 0;
                for param in self.optimizer.params:
                    dtParam, noiX, mse = self.dpcrs[self._shareId].dpRelease(param.grad * (-lr))
                    if incMode:
                        param.add_(noiX)
                    else:
                        param.set_(dtParam + 0)
                    self._shareId += 1;
        return self.optimizer.params;

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none)

    def __repr__(self):
        return self.optimizer.__repr__()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    @property
    def params(self):
        return self.optimizer.params
