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

# This code is for DPCR model of ABCRG (advanced binary-indexed-tree based DPCR model for Gaussian mechanism) Method[1]
# For more details see our work "A Federated Learning Framework Based on Differential Privacy Continuous Data Release"
##########################################################################
import numpy as np
import math
from dpcrpy.utils.bitOps import lowbit, lb, kLAN
from dpcrpy.framework.dpcrMech import DpcrMech


class ABCRG(DpcrMech):
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
        self.mse = 0;
        self.sNoi = 0;
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
        self.t += 1;
        # from dev.FLDPCR.utils.check import checkNorm
        # print(checkNorm(x));
        lp = lb(lowbit(self.t))
        cof2 = self.getCof2(self.t);
        cof = np.sqrt(cof2);
        if self.t % 2 == 0:
            cErr = cof * (self.buff[lp] + x);
            self.buff[lp] = None;
        else:
            cErr = cof * x + self.noiMech.genNoise();
        tmp1 = kLAN(self.t);
        rou = 1.0 - cof2;
        while tmp1 < self.T:
            j = lb(lowbit(tmp1))
            cofTmp1 = self.getCof2(tmp1);
            if self.buff[j] is None:
                self.buff[j] = self.noiMech.genNoise() / math.sqrt(cofTmp1);
            rou -= cofTmp1;
            self.buff[j] += x;
            tmp1 = kLAN(tmp1)
        while len(self.stk) > 0:
            if self.stk[-1][0] <= lp:
                self.stk.pop();
            else:
                break;
        if rou > 1e-8:
            leftCof = np.sqrt(rou);
            xErr2 = x + self.noiMech.genNoise() / leftCof;
            v2 = xErr2 + self.sNoi - (self.stk[-1][2] if len(self.stk) > 0 else 0);
            D2 = self.mse + 1 / rou - (self.stk[-1][1] if len(self.stk) > 0 else 0);
            v1 = cErr / cof;
            D1 = 1 / cof2;
            alpha = D2 / (D1 + D2);
            self.mse = D1 * D2 / (D1 + D2) + (self.stk[-1][1] if len(self.stk) > 0 else 0);
            self.sNoi = alpha * v1 + (1 - alpha) * v2 + (self.stk[-1][2] if len(self.stk) > 0 else 0);
        else:
            self.mse = 1 / cof2 + (self.stk[-1][1] if len(self.stk) > 0 else 0);
            self.sNoi = cErr / cof + (self.stk[-1][2] if len(self.stk) > 0 else 0);
        # print(checkNorm(self.sNoi));
        self.stk.append((lp, self.mse, self.sNoi))
        mse = self.noiMech.getMse() * self.mse;
        return (self.sNoi, mse)
