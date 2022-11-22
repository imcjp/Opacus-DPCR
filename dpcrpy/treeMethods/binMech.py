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

# This code is for DPCR model of Binary tree[1]
# [1] T.-H. H. Chan, E. Shi, and D. Song, “Private and continual release of statistics,” ACM Transactions on Information and System Security (TISSEC), vol. 14, no. 3, pp. 1–24, 2011
##########################################################################
import numpy as np
from dpcrpy.utils.bitOps import lowbit, kLAN, kLRN, lb
from dpcrpy.framework.dpcrMech import DpcrMech


class BinMech(DpcrMech):
    def __init__(self, kOrder=0, noiMech=None, isInit=True):
        self.T = 2 ** kOrder;
        self.kOrder = kOrder;

        self.setNoiMech(noiMech)
        if isInit:
            self.init();

    def init(self):
        self.t = 0;
        self.lastRes = [0] * (self.kOrder + 1);
        self.lastMse = [0] * (self.kOrder + 1);
        self.noiAlphan = [None] * (self.kOrder + 1);
        return self

    def getL1Sens(self):
        return self.kOrder + 1;

    def getL2Sens(self):
        return np.sqrt(self.kOrder + 1);

    def dpRelease(self, x):
        self.t += 1;
        tmp1 = self.t;
        while tmp1 <= self.T:
            j = lb(lowbit(tmp1))
            if self.noiAlphan[j] is None:
                self.noiAlphan[j] = self.noiMech.genNoise()
            self.noiAlphan[j] += x;
            tmp1 = kLAN(tmp1)
        i = lb(lowbit(self.t))
        res = self.noiAlphan[i];
        mse = self.noiMech.getMse();
        tmp1 = kLRN(self.t);
        if tmp1 > 0:
            j = lb(lowbit(tmp1))
            res += self.lastRes[j];
            mse += self.lastMse[j];
        self.lastRes[i] = res;
        self.lastMse[i] = mse;
        self.noiAlphan[i] = None;
        return (res, mse)
