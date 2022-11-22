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
# This script is used to perform experiments for FL-DPCR
##########################################################################
from fldpcr.server import Server

def runFLDPCR(configs):
    ## Configuration Information
    global_config = configs["global_config"]
    data_config = configs["data_config"]
    fed_config = configs["fed_config"]
    optim_config = configs["optim_config"]
    init_config = configs["init_config"]
    model_config = configs["model_config"]
    dp_config = configs["dp_config"]
    dpcr_model = configs["dpcr_model"]

    message = "\n[WELCOME] Unfolding configurations...!"
    print(message);
    for key in configs:
        print({key:configs[key]});

    ## Start Run
    # initialize federated learning
    central_server = Server(model_config, global_config, data_config, init_config, fed_config, optim_config,dp_config,dpcr_model)
    central_server.setup()

    # do federated learning
    central_server.fit()

    message = "...done all learning process!\n...exit program!"
    print(message);

if __name__ == '__main__':
    # Set experimental parameters
    config = {
        'global_config': {
            # If usingDP = True, private federated learning is achieved using differential privacy.
            # The communication round number is indicated by fed_config.R
            # Otherwise, the federated learning is without privacy.
            'usingDP': True,
            'device': 'cuda',   #GPU or CPU
            'is_mp': False  # If True, training can be done in parallel using multiple threads
        },
        'data_config': { # dataset infomation.
            'data_path': 'data',
            'dataset_name': 'MNIST',
            'num_shards': 200,
        },
        'fed_config': {
            'C': 1, #Percentage of participants participating in training per communication round
            'K': 20,    #Number of participants
            'R': 20000, #Maximum communication rounds. It may be less than actual communication rounds due to exhaustion of privacy budget.
            'E': 100,   #Internal iteration number
            'sample_rate': 0.1, #Sampling rate of each iteration
        },
        'optim_config': {
            'lr': 0.1  #Learning rate
        },
        'init_config': {
            'init_type': 'xavier',
            'init_gain': 1.0,
            'gpu_ids': [0]
        },
        'model_config': {
            'name': 'CNN',  #The available models see the file models/__init__.py
            'args': {   #Arguments needed to construct the parameter model
                'in_channels': 1,
                'hidden_channels': 8,
                'num_hiddens': 32,
                'num_classes': 10
            }
        },
        'dp_config': {
            #Privacy Parameters for (epsilon, delta)-DP
            'epsilon': 1,
            'delta': 0.000333,
            'max_norm': 1.0,
            # If 'isFixedClientT' is True, the iteration number of participants is indicated by using 'ClientT' and sigma is calculated by the private budget and 'ClientT'.
            # Otherwise, use 'sigma' to indicate the amount of noise added by the participant, and 'ClientT' is calculated based on the privacy budget and 'sigma'
            'isFixedClientT': True,
            'clientT': 4096,    #Only works for isFixedClientT = True
            'sigma': 1    #Only works for isFixedClientT = False
        },
        'dpcr_model': {
            'name': 'ABCRG',  #Available DPCR models: DPFedAvg (without DPCR), SimpleMech, TwoLevel, BinMech, FDA, BCRG, ABCRG
            'args':{
                'kOrder': 12    #Only works for TwoLevel, BinMech, FDA, BCRG and ABCRG; Ignored for DPFedAvg and SimpleMech
            }
        }
    }

    runFLDPCR(config)