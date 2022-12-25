# ==============================================================================
# Copyright (c) 2022 The PersFormer Authors. All Rights Reserved.
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
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import dtype
from networks.Unet_parts import Down, Up

# overall network

class SegmentHead(nn.Module):
    def __init__(self, channels=128):
        super(SegmentHead, self).__init__()
        self.down1 = Down(channels, channels*2)     # Down(128, 256)
        self.down2 = Down(channels*2, channels*4)   # Down(256, 512)
        self.down3 = Down(channels*4, channels*4)   # Down(512, 512)
        self.up1 = Up(channels*8, channels*2)       # Up(1024, 256)
        self.up2 = Up(channels*4, channels)         # Up(512, 128)
        self.up3 = Up(channels*2, channels)         # Up(256, 128)
        self.segment_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, input):
        x1 = self.down1(input)                      # 128 -> 256
        x2 = self.down2(x1)                         # 256 -> 512
        x3 = self.down3(x2)                         # 512 -> 512
        x_out = self.up1(x3, x2)                    # 512+512 -> 256
        x_out = self.up2(x_out, x1)                 # 256+256 -> 128
        x_out = self.up3(x_out, input)              # 128+128 ->128
        pred_seg_bev_map = self.segment_head(x_out) # 128 -> 1

        return pred_seg_bev_map
