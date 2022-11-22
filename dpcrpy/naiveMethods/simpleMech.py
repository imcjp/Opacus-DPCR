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

# This code is for simple DPCR model of directly accumulating noise
##########################################################################
import numpy as np
from dpcrpy.framework.dpcrMech import DpcrMech
class SimpleMech(DpcrMech):
    def __init__(self,T=1,noiMech=None,isInit=True):
        self.T=T;
        self.setNoiMech(noiMech)
        if isInit:
            self.init();
        
    def getL1Sens(self):
        return self.T;
    
    def getL2Sens(self):
        return np.sqrt(self.T);
    
    def init(self):
        self.t=0;
        return self
        
    def dpRelease(self,x):
        if self.t==0:
            self.sNoi = [self.noiMech.genNoise() for i in range(self.T)];
        for i in range(self.t,self.T):
            self.sNoi[i]+=x;
        res=self.sNoi[self.t];
        self.t+=1;
        mse=self.noiMech.getMse();
        return (res,mse)
