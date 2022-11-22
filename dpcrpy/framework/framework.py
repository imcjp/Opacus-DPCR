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
# The code is a framework for DPCR and supports setting various DPCR models.
##########################################################################

import warnings
class dpCrFw:
    def __init__(self,genBlk,newBlkCallback=None):
        self.genBlk=genBlk;
        self.blkId=0;
        self.t=0;
        self.blk=None;
        self.j=0;
        self.lastRs=0;
        self.lastMse=0;
        self.cumSum=0;
        self.cumMse=0;
        self.lastRes=0;
        self.newBlkCallback=newBlkCallback;
    def initRelease(self,x):
        if self.t==0:
            self.lastRs=x
            self.lastRes = x
        else:
            warnings('InitRelease should be executed before the first release. The execution is aborted!')

    def dpRelease(self,x):
        if (self.blk is None) or self.blk.size()==self.j:
            self.blkId+=1;
            self.blk=self.genBlk(self.blkId);
            self.j=0;
            self.cumSum+=self.lastRs;
            self.cumMse+=self.lastMse;
            if not self.newBlkCallback is None:
                self.newBlkCallback(self.blkId,self.blk,self.cumMse,self.cumSum);
        (self.lastRs,self.lastMse)=self.blk.dpRelease(x);
        res=self.cumSum+self.lastRs
        noiX=res-self.lastRes;
        self.lastRes=res;
        if self.lastMse is None:
            mse=None
        else:
            mse=self.cumMse+self.lastMse;
        self.t+=1;
        self.j+=1;
        return (res,noiX,mse)

    def __str__(self):
        res=(f'已经发布了{self.t}次\n');
        res+=(f'当前块是{self.blkId}块，该块已经发布了{self.j}次')
        return res;