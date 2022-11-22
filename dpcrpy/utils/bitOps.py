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

# This code provides methods for operating on Binary Indexed Tree (BIT)
# For more details see our work "A Federated Learning Framework Based on Differential Privacy Continuous Data Release"

##########################################################################

import numpy as np
def bitLen(x):
    bn=list(bin(x))
    return bn.count('1')
def lowbit(x):
    x=int(x)
    return x & -x
def kLAN(x,k=1):
    for i in range(k):
        x+=lowbit(x)
    return x
def kLRN(x,k=1):
    for i in range(k):
        x-=lowbit(x)
    return x
def lb(x):
    return int(np.log2(x))