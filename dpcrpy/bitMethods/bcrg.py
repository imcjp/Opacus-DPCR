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

# This code is for DPCR model of BCRG (binary-indexed-tree based DPCR model for Gaussian mechanism) Method[1]
# For more details see our work "A Federated Learning Framework Based on Differential Privacy Continuous Data Release"
##########################################################################
import numpy as np
from dpcrpy.utils.bitOps import lowbit, lb, kLAN
from dpcrpy.framework.dpcrMech import DpcrMech


class BCRG(DpcrMech):
    def __init__(self, kOrder=1, noiMech=None, isInit=True):
        self.T = 2 ** kOrder - 1;
        self.kOrder = kOrder;
        (self.alphaArr, _) = self.genAlphaGauss(self.kOrder)
        self.setNoiMech(noiMech)
        if isInit:
            self.init();

    def init(self):
        self.t = 0;
        self.stk = [];
        self.buff = [None] * (self.kOrder);
        return self

    def getL1Sens(self):
        sen = 0;
        for i in range(self.kOrder):
            h = self.alphaArr[i];
            sen = max(sen, np.sqrt(h) + np.sqrt(1 - h) * sen)
        return sen;

    def getL2Sens(self):
        return 1;

    def genAlphaGauss(self, k):
        h = 1;
        t = 2;
        alphan = [0] * k;
        alphan[0] = 1;
        for i in range(1, k):
            alphan[i] = t ** (1 / 2) / (t ** (1 / 2) + h ** (1 / 2));
            h += (t ** (1 / 2) + h ** (1 / 2)) ** 2;
            t += t;
        return (alphan, h)

    def getCof2(self, k):
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
        # from dev.FLDPCR.utils.check import checkNorm
        # print(checkNorm(x));
        self.t += 1;
        lp = lb(lowbit(self.t))
        cof2 = self.getCof2(self.t);
        cof = np.sqrt(cof2);
        if self.t % 2 == 0:
            cErr = cof * (self.buff[lp] + x);
            self.buff[lp] = None;
        else:
            cErr = cof * x + self.noiMech.genNoise();
        tmp1 = kLAN(self.t);
        while tmp1 < self.T:
            j = lb(lowbit(tmp1))
            if self.buff[j] is None:
                cofTmp1 = np.sqrt(self.getCof2(tmp1));
                self.buff[j] = self.noiMech.genNoise() / cofTmp1;
            self.buff[j] += x;
            tmp1 = kLAN(tmp1)

        while len(self.stk) > 0:
            if self.stk[-1][0] <= lp:
                self.stk.pop();
            else:
                break;
        mse = 1 / cof2 + (self.stk[-1][1] if len(self.stk) > 0 else 0);
        sNoi = cErr / cof + (self.stk[-1][2] if len(self.stk) > 0 else 0);
        # print(checkNorm(sNoi));
        self.stk.append((lp, mse, sNoi))
        mse *= self.noiMech.getMse()
        return (sNoi, mse)
