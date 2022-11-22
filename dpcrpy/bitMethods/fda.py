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

# This code is for DPCR model of FDA Method[1]
# [1]  J. Cai, W. U. Yingjie, and X. Wang, “Method based on matrix mechanism for differential privacy continual data release,” Journal of Frontiers of Computer Science and Technology, 2016.
##########################################################################
from dpcrpy.utils.bitOps import lowbit, lb
from dpcrpy.framework.dpcrMech import DpcrMech


class FDA(DpcrMech):
    def __init__(self, kOrder=1, noiMech=None, isInit=True):
        self.T = 2 ** kOrder - 1;
        self.kOrder = kOrder;
        (self.alphaArr, _) = self.genAlphaLap(self.kOrder)
        self.setNoiMech(noiMech)
        if isInit:
            self.init();

    def init(self):
        self.t = 0;
        self.stk = [];
        self.s = 0;
        return self

    def getL1Sens(self):
        return 1;

    def getL2Sens(self):
        return 1;

    def genAlphaLap(self, k):
        h = 1;
        t = 2;
        alphan = [0] * k;
        alphan[0] = 1;
        for i in range(1, k):
            alphan[i] = t ** (1 / 3) / (t ** (1 / 3) + h ** (1 / 3));
            h += (t ** (1 / 3) + h ** (1 / 3)) ** 3;
            t += t;
        return (alphan, h)

    def getCof(self, k):
        l = len(self.alphaArr)
        x = 0;
        cof = 1
        for i in range(l):
            r = k % 2
            if x == 0:
                if r == 1:
                    cof *= self.alphaArr[i]
                    x = 1
            else:
                if r == 0:
                    cof *= 1 - self.alphaArr[i]
            k //= 2;
        return cof;

    def dpRelease(self, x):
        self.t += 1;
        self.s = self.s + x;
        lp = lb(lowbit(self.t))
        while len(self.stk) > 0:
            if self.stk[-1][0] <= lp:
                self.stk.pop();
            else:
                break;
        cof = self.getCof(self.t)
        cof2 = cof * cof;
        c = self.s - (self.stk[-1][1] if len(self.stk) > 0 else 0)
        cErr = cof * c + self.noiMech.genNoise();
        mse = 1 / cof2 + (self.stk[-1][2] if len(self.stk) > 0 else 0);
        sNoi = cErr / cof + (self.stk[-1][3] if len(self.stk) > 0 else 0);
        self.stk.append((lp, self.s, mse, sNoi))
        mse *= self.noiMech.getMse()
        return (sNoi, mse)
