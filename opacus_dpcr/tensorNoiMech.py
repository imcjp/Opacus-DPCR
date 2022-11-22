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
# It succeeds GaussNoiMech for supporting tensor.
##########################################################################
from dpcrpy.framework.noiMech import GaussNoiMech
import torch

class TensorGaussNoiMech(GaussNoiMech):
    def __init__(self, epsilon=None, delta=0, sens=1, sigma0=None, tsSize=(1,), device=None):
        GaussNoiMech.__init__(self, epsilon=epsilon, delta=delta, sens=sens, sigma0=sigma0)
        self.tsSize = tsSize
        self.device = device

    def genNoise(self, shape=None):
        if shape is None:
            rd = torch.normal(
                mean=0,
                std=self.sigma,
                size=list(self.tsSize),
                device=self.device
            )
            return -rd;
        rd = torch.normal(
            mean=0,
            std=self.sigma,
            size=list(shape) + list(self.tsSize),
            device=self.device
        )
        return -rd;
