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

import os
import numpy as np
import pickle
import dpcrpy as dcr
import time
theTic = 0
def tic():
    global theTic
    theTic = time.perf_counter()
    return theTic

def toc(timeObj=None):
    if timeObj is not None:
        return time.perf_counter()-timeObj;
    else:
        return time.perf_counter()-theTic;

# Performing your DPCR experiment
def runDPCR(configs):
    global_config=configs["global_config"]
    alg_config=configs["alg_config"]
    noi_config=configs["noise_config"]
    print("The setting information is as follows:");
    for key in configs:
        print({key:configs[key]});
    noiMech = eval(f'dcr.noiMech.{noi_config["name"]}')(epsilon=noi_config['args']['epsilon'],delta=noi_config['args']['delta'])
    if alg_config['name']=='SimpleMech':
        dpcrMech=dcr.gen(alg_config['name']).setNoiMech(noiMech)
    else:
        dpcrMech=dcr.gen(alg_config['name'], alg_config['args']).setNoiMech(noiMech)
    genBk = lambda x: dpcrMech.init()
    maxTimes= global_config['maxTimes']
    if maxTimes==-1:
        maxTimes=np.Inf;
    dataPath = os.path.join('../data',global_config['dataset_name']+'.plk')
    pkl_file = open(dataPath, 'rb')
    cnt = pickle.load(pkl_file)
    pkl_file.close()
    vn = cnt;
    rs = dcr.dpCrFw(genBk)
    trueRes=0;
    allMse=0;
    allTMse=0;
    runned=0;
    tic();
    for v in vn:
        if runned==maxTimes:
            break;
        trueRes+=v;
        (res, noiX, tMse) = rs.dpRelease(v);
        mse=(trueRes-res)**2;
        allMse += mse;
        allTMse += tMse;
        if global_config['isOutputDetail']:
            print(f'The {runned+1}-th release results are as follows:')
            print(f'\tThe received increment is {v:.2f}, the released increment is {noiX:.2f}')
            print(f'\tThe true accumulation is {trueRes:.2f}, the released results are{ res:.2f}')
            print(f'\tThe theoretical std is {np.sqrt(tMse):.2f}, the actual std is {np.sqrt(mse):.2f}')
            print(f'\tThe running time is {toc()}s')
        runned+=1;
    runTime=toc();
    rmse=np.sqrt(allMse/runned)
    tRmse=np.sqrt(allTMse/runned)

    print(f'Summary of the experiment:')
    print(f'\tThe actual overall rmse is {rmse:.2f}')
    print(f'\tThe theoretical overall rmse is {tRmse:.2f}')
    print(f'\tTotal running time is {runTime}s')
    print(f'\tActual release {runned} times')
    print(f'\tThe maximum release count (T) for each Block is {dpcrMech.size()} times')
    print(f'The information of the noise mechanism is as follows:')
    print(noiMech)


if __name__ == '__main__':
    # Set experimental parameters
    config={
        'global_config': {
            'isOutputDetail': True,   #设Set to True and you will see more details of the release
            'maxTimes': 16000,   #The maximum number of releases. If set to -1, it means no limit on the release times
            'dataset_name': 'pageriew'  #Available datasets: cfc, trip1, pageriew
        },
        'alg_config': {
            'name': 'FDA',    #Available algorithms: SimpleMech, TwoLevel, BinMech, FDA, BCRG, ABCRG
            'args': {
                'kOrder': 14    #The scale parameter of the algorithm k. This parameter is not valid for SimpleMech.
            }
        },
        'noise_config': {
            'name': 'GaussNoiMech',     #Available noise mechanisms: GaussNoiMech and LapNoiMech
            'args': {       #Privacy Parameters (epsilon, delta)-DP
                'epsilon': 1,
                'delta': 1e-04
            }
        }
    }
    runDPCR(config)