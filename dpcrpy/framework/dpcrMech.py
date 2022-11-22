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

# Superclass for DPCR models
##########################################################################
class DpcrMech:
    def __init__(self):
        pass

    def setNoiMech(self, noiMech):
        self.noiMech = noiMech;
        if self.noiMech is not None:
            if self.noiMech.getSensType() == 1:
                self.noiMech.setSens(self.getL1Sens());
            elif self.noiMech.getSensType() == 2:
                self.noiMech.setSens(self.getL2Sens());
        return self

    def init(self):
        pass

    def getL1Sens(self):
        pass

    def getL2Sens(self):
        pass

    def dpRelease(self, x):
        pass

    def size(self):
        return self.T
