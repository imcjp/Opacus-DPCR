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
# Server for FL-DPCR (Collaborator)
##########################################################################
import copy
import gc
import numpy as np
import torch

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from collections import OrderedDict
from fldpcr.utils.utils import init_net,create_datasets
from fldpcr.priClient import PriClient
from fldpcr.flClient import FLClient
from opacus.validators import ModuleValidator
import models

def printOnRunning(info=None):
    if isinstance(info, dict):
        stage = info['stage']
        strInfo = info['info']
        print(f'[{stage}]\t{strInfo}')
    else:
        print(info)

def afterEachIteration(info):
    round = info['round']
    test_loss = info['loss']
    test_accuracy = info['accuracy']
    epsUseds = info['epsUseds']
    message = f"[Round: {str(round).zfill(4)}] Evaluate global model's performance...!\
                    \n\t[Server] ...finished evaluation!\
                    \n\t=> Loss: {test_loss:.4f}\
                    \n\t=> Accuracy: {100. * test_accuracy:.2f}%"
    for i in range(len(epsUseds)):
        if i % 5 == 0:
            message += "\n"
        message += f"\t=> EpsilonUsed for client {str(i + 1).zfill(4)}: {epsUseds[i]:.4f}";
    message += "\n"
    print(message);
        
        
class Server(object):

    def __init__(self, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={},
                 optim_config={}, dp_config={}, dpcr_model={}):
        self.clients = None
        self._round = 0
        self.modelName = model_config["name"];
        self.model = models.gen(self.modelName, model_config["args"] if "args" in model_config else None)
        if global_config['usingDP'] and not ModuleValidator.is_valid(self.model):
            self.model = ModuleValidator.fix(self.model)

        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]
        if global_config["usingDP"]:
            self.ClientClass = PriClient
        else:
            self.ClientClass = FLClient

        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]

        self.init_config = init_config

        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.sample_rate = fed_config["sample_rate"]

        self.criterion = 'torch.nn.CrossEntropyLoss'
        self.optimizer = 'torch.optim.SGD'
        self.optim_config = optim_config
        self.dp_config = dp_config
        self.dpcr_model = dpcr_model

    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        init_net(self.model, **self.init_config)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        printOnRunning(message);
        del message;
        gc.collect()
        # split local dataset for each client
        local_datasets, test_dataset = create_datasets(self.data_path, self.dataset_name, self.num_clients,
                                                       self.num_shards, True)

        # assign dataset to each client
        self.clients = self.create_clients(local_datasets)

        # prepare hold-out dataset for evaluation
        self.data = test_dataset
        self.dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

        # configure detailed settings for client upate and 
        self.setup_clients(
            sample_rate=self.sample_rate,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config, dp_config=self.dp_config,
            dpcr_model=self.dpcr_model
        )

        # send the model skeleton to all clients
        self.initClientModel()

    def create_clients(self, local_datasets):
        """Initialize each Client instance."""
        clients = []
        self.total_data_size = 0;
        for k, dataset in enumerate(local_datasets):
            client = self.ClientClass(client_id=k, local_data=dataset, device=self.device)
            clients.append(client)
            self.total_data_size += client.getDatasetSize()
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        printOnRunning(message);
        del message;
        gc.collect()
        return clients

    def mp_setup_clients(self, client_index, **client_config):
        self.clients[client_index].setup(**client_config)

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in enumerate(self.clients):
            client.setup(**client_config)
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        printOnRunning(message);
        del message;
        gc.collect()

    def initClientModel(self):
        """Send the updated global model to selected/all clients."""
        assert (self._round == 0) or (self._round == self.num_rounds)

        for client in self.clients:
            client.setModel(copy.deepcopy(self.model))

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
        printOnRunning(message);
        del message;
        gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round

            for client in self.clients:
                client.setModel(copy.deepcopy(self.model))

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            printOnRunning(message);
            del message;
            gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            for idx in sampled_client_indices:
                self.clients[idx].model = copy.deepcopy(self.model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            printOnRunning(message);
            del message;
            gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        printOnRunning(message);
        del message;
        gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        availableClientIds = []
        for i in range(self.num_clients):
            if not self.clients[i].isStopped():
                availableClientIds.append(i);
        if len(availableClientIds) > num_sampled_clients:
            sampled_client_indices = sorted(
                np.random.choice(a=availableClientIds, size=num_sampled_clients, replace=False).tolist())
        else:
            sampled_client_indices = availableClientIds;
        return sampled_client_indices

    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        printOnRunning(message);
        del message;
        gc.collect()

        selected_total_size = 0
        param_dict = self.model.state_dict()

        for idx in sampled_client_indices:
            self.clients[idx].client_update(param_dict)

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        printOnRunning(message);
        del message;
        gc.collect()

    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        printOnRunning(message, flush=True);
        del message;
        gc.collect()

        param_dict = self.model.state_dict()
        self.clients[selected_index].client_update(param_dict)

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated!"
        printOnRunning(message, flush=True);
        del message;
        gc.collect()

    def aggregateAfterAccumulation(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        printOnRunning(message);
        del message;
        gc.collect()
        # 如果对coefficients归一化，则保持其与FedAvg一样
        coefficients = coefficients / np.sum(coefficients)

        averaged_weights = OrderedDict()

        for it, idx in enumerate(sampled_client_indices):
            local_weights = self.clients[idx].lastLocalModelChange
            for key in self.model.state_dict().keys():
                client_weight = coefficients[it] * local_weights[key]
                if it == 0:
                    averaged_weights[key] = client_weight
                else:
                    averaged_weights[key] += client_weight
        modelDict = self.model.state_dict()
        for key in self.model.state_dict().keys():
            if not modelDict[key].dtype==averaged_weights[key].dtype:
                printOnRunning(f'{key} 类型不匹配，执行强制转换')
                averaged_weights[key]=averaged_weights[key].type_as(modelDict[key])
            modelDict[key] += averaged_weights[key]
        # self.model.load_state_dict(averaged_weights)
        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        printOnRunning(message);
        del message;
        gc.collect()

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()
        if len(sampled_client_indices) == 0:
            return False;
        printOnRunning(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            self.param_dict = self.model.state_dict()
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
        else:
            self.update_selected_clients(sampled_client_indices)

        mixing_coefficients = [self.clients[idx].getDatasetSize() / self.total_data_size for idx in
                               sampled_client_indices]
        self.aggregateAfterAccumulation(sampled_client_indices, mixing_coefficients)
        # average each updated model parameters of the selected clients and update the global model
        self.aggregateAfterAccumulation(sampled_client_indices, mixing_coefficients)

        return True;

    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)
        return test_loss, test_accuracy

    def fit(self):
        """Execute the whole process of the federated learning."""
        for r in range(self.num_rounds):
            self._round = r + 1
            isTrain = self.train_federated_model()
            if not isTrain:
                printOnRunning("Privacy exhausted, training over!")
                break;
            test_loss, test_accuracy = self.evaluate_global_model()

            epsUseds=[];
            for i in range(len(self.clients)):
                epsUseds.append(self.clients[i].epsUsed)
            info = {'name': f"[{self.dataset_name}]_{self.modelName}", 'round': self._round,
                    'loss': test_loss, 'accuracy': test_accuracy, 'epsUseds': epsUseds}
            afterEachIteration(info);
            gc.collect()
        self.transmit_model()
