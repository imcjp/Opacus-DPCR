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
# The participant's client for federated learning with DP
##########################################################################
import gc
import torch

import dpcrpy
from torch.utils.data import DataLoader
from opacus_dpcr.dpcr_engine import DPCREngine
from opacus import PrivacyEngine
from opacus_dpcr.dpcrOptimizer import DPCROptimizer
from fldpcr.utils.rdpAccount import getMinSigma,getClientT
from collections import OrderedDict

class PriClient(object):

    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self._pri_model = None
        self._pri_optimizer = None
        self._pri_train_loader = None
        ###############################
        self.epsUsed = 0

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        batch_size=client_config["sample_rate"]*self.data.tensors[0].size(0);
        batch_size=max(round(batch_size),1)
        self.dataloader = DataLoader(self.data, batch_size=batch_size, pin_memory=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.criterion = eval(self.criterion)()
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]
        self.accEpoch = 0;
        self.max_per_sample_grad_norm = float(client_config["dp_config"]["max_norm"])
        self.epsilon = float(client_config["dp_config"]["epsilon"])
        self.delta = float(client_config["dp_config"]["delta"])

        if client_config["dp_config"]['isFixedClientT']:
            self.maxAllowedIter=client_config["dp_config"]['clientT']
            sample_rate=1 / len(self.dataloader);
            self.sigma = getMinSigma(sample_rate=sample_rate, num_steps=self.maxAllowedIter, delta=self.delta, requireEps=self.epsilon);
            print(f'Under maximum iteration number = {self.maxAllowedIter} and sample_rate = {sample_rate}, the sigma of F{1+self.id} is {self.sigma}.')
        else:
            self.sigma = float(client_config["dp_config"]["sigma"])
            sample_rate=1 / len(self.dataloader);
            self.maxAllowedIter = getClientT(sample_rate=sample_rate, maxEps=self.epsilon, delta=self.delta, sigma=self.sigma )
            print(f'Under sigma = {self.sigma} and sample_rate = {sample_rate}, the maximum iteration number of F{1+self.id} is {self.maxAllowedIter}.')

        if client_config["dpcr_model"]['name'] == 'DPFedAvg':
            self.privacy_engine = PrivacyEngine(secure_mode=False);
        else:
            if client_config["dpcr_model"]['name']== 'SimpleMech':
                genBk = lambda x: dpcrpy.gen(client_config["dpcr_model"]['name'])
            else:
                genBk = lambda x: dpcrpy.gen(client_config["dpcr_model"]['name'],client_config["dpcr_model"]['args'])
            self.privacy_engine = DPCREngine(secure_mode=False, dpcrGenerator=genBk)

    def client_update(self, param_dict):
        """Update local model using local dataset."""
        xx = True
        with torch.no_grad():
            if xx:
                self._pri_model._modules['_module'].load_state_dict(param_dict)
        self._pri_model.train()
        losses = []
        localEpochAllowed=min(self.maxAllowedIter-self.accEpoch,self.local_epoch)
        _batch_idx=0;
        if self.id==0:
            print(f'Local updates {localEpochAllowed} times...')
        while True:
            for _, (data, target) in enumerate(self._pri_train_loader):
                if _batch_idx == localEpochAllowed:
                    break;
                data, target = data.float().to(self.device), target.long().to(self.device)
                self._pri_optimizer.zero_grad()
                if data.size(0)>0:
                    output = self._pri_model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    losses.append(loss.item())
                else:
                    losses.append(0.0)
                if isinstance(self._pri_optimizer, DPCROptimizer):
                    self._pri_optimizer.step(incMode=True)
                else:
                    self._pri_optimizer.step()
                self.accEpoch += 1
                _batch_idx+=1;
                if self.device == "cuda": torch.cuda.empty_cache()
            if _batch_idx == localEpochAllowed:
                break;
        self.epsUsed, best_alpha = self.privacy_engine.accountant.get_privacy_spent(
            delta=self.delta
        )

        self._dt_local_weight=OrderedDict()
        local_weights = self._pri_model.state_dict()
        for key in param_dict.keys():
            self._dt_local_weight[key]=local_weights['_module.' + key]-param_dict[key];
        gc.collect()

    def getModel(self):
        """Local model setter for passing globally aggregated model parameters."""
        return self._pri_model

    def setModel(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        if self._pri_model is None:
            optimizer = eval(self.optimizer)(model.parameters(), **self.optim_config)
            self._pri_model, self._pri_optimizer, self._pri_train_loader = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=self.dataloader,
                noise_multiplier=self.sigma,
                max_grad_norm=self.max_per_sample_grad_norm
            )
        else:
            with torch.no_grad():
                self._pri_model._modules['_module'].load_state_dict(model.state_dict())

    def isStopped(self):
        return self.accEpoch >= self.maxAllowedIter

    @property
    def lastLocalModelChange(self):
        return self._dt_local_weight

    def getDatasetSize(self):
        return self.data.tensors[0].size(0);