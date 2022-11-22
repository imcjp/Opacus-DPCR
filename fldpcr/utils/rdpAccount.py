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
# The tools for Calibrating Privacy
##########################################################################
from opacus.accountants.analysis import rdp as privacy_analysis
import json

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 201)] + list(range(22, 500))
callBak={};

def loadBakedObj(key):
    str=json.dumps(key);
    if str in callBak:
        return callBak[str]
    return None

def bakObj(key,val):
    str=json.dumps(key);
    if str not in callBak:
        callBak[str]=val;

def getPrivacySpentWithFixedNoise(
        sample_rate, num_steps, delta: float,sigma=1.0, alphas = None
):
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    noise_multiplier=sigma;
    rdp = privacy_analysis.compute_rdp(
                q=sample_rate,
                noise_multiplier=noise_multiplier,
                steps=num_steps,
                orders=alphas,
            )
    eps, best_alpha = privacy_analysis.get_privacy_spent(
        orders=alphas, rdp=rdp, delta=delta
    )
    return float(eps), float(best_alpha)

def epochAllowed(sample_rate, stepsHasRunned,requiredSteps, maxEps, delta: float,sigma=1.0, alphas = None):
    epsUsed, best_alpha = getPrivacySpentWithFixedNoise(sample_rate, stepsHasRunned+requiredSteps, delta, sigma=sigma, alphas = alphas);
    if maxEps<epsUsed:
        minStep=0
        maxStep=requiredSteps-1;
        while minStep<maxStep:
            midStep=(minStep+maxStep+1)//2
            epsUsed, best_alpha = getPrivacySpentWithFixedNoise(sample_rate, stepsHasRunned+midStep, delta, sigma=sigma, alphas = alphas);
            if maxEps<epsUsed:
                maxStep=midStep-1;
            else:
                minStep=midStep;
        return minStep
    else:
        return requiredSteps;

def getClientT(sample_rate, maxEps, delta: float,sigma=1.0, alphas = None):
    keyArr=['getMinSigma',sample_rate, maxEps, delta, sigma, alphas];
    obj=loadBakedObj(keyArr);
    if obj is not None:
        return obj
    minStep=0;
    maxStep=1;
    while getPrivacySpentWithFixedNoise(sample_rate, maxStep, delta, sigma=sigma, alphas = alphas)[0]<=maxEps:
        minStep=maxStep;
        maxStep*=2;
    maxStep = maxStep - 1;
    while minStep<maxStep:
        midStep=(minStep+maxStep+1)//2
        epsUsed, best_alpha = getPrivacySpentWithFixedNoise(sample_rate, midStep, delta, sigma=sigma, alphas = alphas);
        if maxEps<epsUsed:
            maxStep=midStep-1;
        else:
            minStep=midStep;
    bakObj(keyArr,minStep)
    return minStep

def getMinSigma(sample_rate, num_steps, delta: float,requireEps, alphas = None):
    keyArr=['getMinSigma',sample_rate, num_steps, delta,requireEps, alphas];
    obj=loadBakedObj(keyArr);
    if obj is not None:
        return obj
    minSigma=0;
    maxSigma = 1;
    while getPrivacySpentWithFixedNoise(sample_rate, num_steps, delta, sigma=maxSigma, alphas = alphas)[0]>requireEps:
        minSigma=maxSigma;
        maxSigma*=2;
    while maxSigma-minSigma>1e-8:
        midSigma=(maxSigma+minSigma)/2;
        eps, best_alpha = getPrivacySpentWithFixedNoise(sample_rate, num_steps, delta, sigma=midSigma, alphas = alphas)
        if eps<=requireEps:
            maxSigma=midSigma;
        else:
            minSigma=midSigma;
    bakObj(keyArr,maxSigma)
    return maxSigma;
