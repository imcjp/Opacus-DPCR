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

# This code is for DPCR model of TwoLevel Method[1]
# [1] T.-H. H. Chan, E. Shi, and D. Song, “Private and continual release of statistics,” ACM Transactions on Information and System Security (TISSEC), vol. 14, no. 3, pp. 1–24, 2011
##########################################################################
import numpy as np
from dpcrpy.framework.dpcrMech import DpcrMech

class TwoLevel(DpcrMech):
    def __init__(self, kOrder=1, noiMech=None, isInit=True):
        self.T = kOrder * kOrder;
        self.B = kOrder;
        self.setNoiMech(noiMech)
        if isInit:
            self.init();

    def getL1Sens(self):
        return 2;

    def getL2Sens(self):
        return np.sqrt(2);

    def init(self):
        self.t = 0;
        self.alpha = 0;
        self.beta = 0;
        self.alphaMse = 0;
        self.betaMse = 0;
        return self;

    def dpRelease(self, x):
        if self.t == 0:
            self.betaBuf = self.noiMech.genNoise();
        self.t += 1;
        self.betaBuf += x;
        self.alpha += self.noiMech.genNoise() + x;
        self.alphaMse += self.noiMech.getMse();
        r = self.t % self.B;
        if r == 0:
            self.beta += self.betaBuf;
            self.betaMse += self.noiMech.getMse();
            self.alpha = 0;
            self.alphaMse = 0;
            self.betaBuf = self.noiMech.genNoise();
        res = self.beta + self.alpha;
        mse = self.alphaMse + self.betaMse;
        return (res, mse)
