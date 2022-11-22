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

from dpcrpy.framework.framework import dpCrFw
import dpcrpy.framework.noiMech as noiMech

## The Methods of Continuous Data Release #################################################
from dpcrpy.naiveMethods.simpleMech import SimpleMech
from dpcrpy.naiveMethods.twoLevel import TwoLevel
from dpcrpy.treeMethods.binMech import BinMech
from dpcrpy.bitMethods.fda import FDA
from dpcrpy.bitMethods.bcrg import BCRG
from dpcrpy.bitMethods.abcrg import ABCRG


## The Tools of Continuous Data Release #################################################
import dpcrpy.utils.bitOps as bitOps

def gen(name,args=None):
    if args is None:
        md= eval(name)();
    else:
        md= eval(name)(**args);
    return md;
