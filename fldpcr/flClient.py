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
# The participant's client for federated learning
##########################################################################
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

class FLClient(object):

    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.epsUsed = 0

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        batch_size=client_config["sample_rate"]*self.data.tensors[0].size(0);
        batch_size=max(round(batch_size),1)
        self.dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.criterion = eval(self.criterion)()
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]
        self.maxAllowedIter=client_config["dp_config"]['clientT']
        if self.maxAllowedIter<0:
            self.maxAllowedIter=np.Inf
        self.accEpoch = 0;


    def client_update(self, param_dict):
        """Update local model using local dataset."""
        xx = True
        with torch.no_grad():
            if xx:
                self._model.load_state_dict(param_dict)
        self._model.train()
        optimizer = eval(self.optimizer)(self._model.parameters(), **self.optim_config)
        localEpochAllowed=min(self.maxAllowedIter-self.accEpoch,self.local_epoch)
        _batch_idx=0;
        while True:
            for _, (data, target) in enumerate(self.dataloader):
                if _batch_idx == localEpochAllowed:
                    break;
                data, target = data.float().to(self.device), target.long().to(self.device)

                optimizer.zero_grad()
                outputs = self._model(data)
                loss = self.criterion(outputs, target)

                loss.backward()
                optimizer.step()

                self.accEpoch += 1
                _batch_idx+=1;

            if _batch_idx == localEpochAllowed:
                break;

        self._dt_local_weight=OrderedDict()
        local_weights = self._model.state_dict()
        for key in param_dict.keys():
            self._dt_local_weight[key]=local_weights[ key]-param_dict[key];
        gc.collect()

    def getModel(self):
        """Local model setter for passing globally aggregated model parameters."""
        # return self._pri_model
        return self._model

    def setModel(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self._model = model

    def isStopped(self):
        return self.accEpoch >= self.maxAllowedIter

    @property
    def lastLocalModelChange(self):
        return self._dt_local_weight

    def getDatasetSize(self):
        return self.data.tensors[0].size(0);