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

# This code is for constructing noise mechanisms, supporting Gaussian mechanism and Laplacian mechanism

##########################################################################
from dpcrpy.utils.dpTools.calibrator_zoo import generalized_eps_delta_calibrator, ana_gaussian_calibrator
from dpcrpy.utils.dpTools.mechanism_zoo import ExactGaussianMechanism, LaplaceMechanism

import numpy as np


class GaussNoiMech:
    def __init__(self, epsilon=None, delta=0, sens=1, sigma0=None):
        self.sens = sens;
        if sigma0 is not None:
            self.epsilon = None;
            self.delta = None;
            self.sigma0 = sigma0;
        else:
            self.epsilon = epsilon;
            self.delta = delta;
            ana_calibrate = ana_gaussian_calibrator()
            mech = ana_calibrate(ExactGaussianMechanism, self.epsilon, self.delta, name='Ana_GM')
            self.sigma0 = mech.params['sigma'];
        self.sigma = sens * self.sigma0;

    def genNoise(self, shape=None):
        if shape is None:
            return np.random.normal(0, self.sigma);
        return np.random.normal(0, self.sigma, shape)

    def getMse(self):
        return self.sigma * self.sigma;

    def getSensType(self):
        return 2;

    def setSens(self, sens):
        self.sens = sens;
        self.sigma = sens * self.sigma0;

    def __str__(self):
        res = '';
        res += f'Gaussian Mechanism with L{self.getSensType()} Sensitivity\n';
        res += (f'Satisfying (ε = {self.epsilon},δ = {self.delta}) - DP and Noise parameter sigma = {self.sigma0}\n');
        res += (f'MSE of Noise is {self.getMse()}。\n')
        return res;

class LapNoiMech:
    def __init__(self, epsilon=None, delta=0, sens=1, b0=None):
        self.sens=sens;
        if b0 is not None:
            self.epsilon=None;
            self.delta=None;
            self.b0=b0;
        else:
            self.epsilon=epsilon;
            self.delta=delta;
            if delta>0:
                calibrate = generalized_eps_delta_calibrator()
                mech = calibrate(LaplaceMechanism, self.epsilon, self.delta, [0.1 / self.epsilon, 10 / self.epsilon], name='Laplace')
                self.b0=mech.params['b'];
            else:
                self.b0=1/self.epsilon;
        self.b=sens*self.b0;
    
    def genNoise(self,shape=()):
        a=np.random.random(shape)-0.5;
        res=self.b*np.sign(a)*np.log(1-np.abs(a)*2);
        return res
    
    def getMse(self):
        return 2*self.b*self.b;
    
    def getSensType(self):
        return 1;
    
    def setSens(self,sens):
        self.sens=sens;
        self.b=sens*self.b0;
        
    def __str__(self):
        res='';
        res+=f'Laplace Mechanism with L{self.getSensType()} Sensitivity\n';
        res+=(f'Satisfying (ε = {self.epsilon},δ = {self.delta}) - DP and Noise parameter b = {self.b0}\n');
        res+=(f'MSE of Noise is {self.getMse()}。\n')
        return res;

class NoNoiMech:
    def __init__(self):
        pass;
    
    def genNoise(self,shape=None):
        return np.zeros(shape)
    
    def getMse(self):
        return 0
    
    def getSensType(self):
        return 0;
    
    def setSens(self,sens):
        pass;
        
    def __str__(self):
        res='No noise mechanism, MSE is 0.\n';
        return res;
