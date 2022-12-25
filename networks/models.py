"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from tools.utils import gen_dx_bx, cumsum_trick, QuickCumsum
from networks.Lane2D import FrontViewPathway,LaneATTHead
from networks.SegHead import SegmentHead


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
    
def make_3d_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x, encoder_feat = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x, encoder_feat

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        # print('x start=',x.shape)

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx==0:
                encoder_feat = x.clone()
            # print('x=',x.shape)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        # print('endpoints[reduction_5]', endpoints['reduction_5'].shape)
        # print('endpoints[reduction_4]', endpoints['reduction_4'].shape)
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x, encoder_feat

    def forward(self, x):
        depth, x, encoder_feat = self.get_depth_feat(x)

        return x, encoder_feat


def make_one_layer(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    if batch_norm:
        layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    else:
        layers = [conv2d, nn.ReLU(inplace=True)]
    return layers
    
def make_one_3d_layer(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False):
    conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    if batch_norm:
        layers = [conv3d, nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True)]
    else:
        layers = [conv3d, nn.ReLU(inplace=True)]
    return layers
    
class BevEncode(nn.Module):
    def __init__(self, inC, outC, num_y_steps):
        super(BevEncode, self).__init__()
        
        
        self.num_y_steps = num_y_steps
        self.anchor_dim = self.num_y_steps + 1

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        # self.up1 = Up(64+256, 256, scale_factor=4)
        # self.up2 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear',
                              # align_corners=True),
            # nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(128, outC, kernel_size=1, padding=0),
        # )
        
        dim_rt_layers = []
        dim_rt_layers += make_one_layer(256, 128, kernel_size=(5, 1), padding=(2, 0), batch_norm=True)#256 instead 11,392
        dim_rt_layers += [nn.Conv2d(128, 64, kernel_size=(5, 1), padding=(2, 0))]
        dim_rt_layers += [nn.Conv2d(64, 32, kernel_size=(5, 1), padding=(2, 0))]
        dim_rt_layers += [nn.Conv2d(32, 16, kernel_size=(5, 1), padding=(2, 0))]
        self.dim_rt = nn.Sequential(*dim_rt_layers)
        
        self.last_layer = nn.Linear(25*25, self.anchor_dim)

    def forward(self, x):
        # print('bev in', x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # print('bev in1', x.shape)

        x1 = self.layer1(x)
        # print('bev layer 1', x1.shape)
        x = self.layer2(x1)
        # print('bev layer 2', x.shape)
        x = self.layer3(x)
        # print('bev layer 3', x.shape)
        x = self.dim_rt (x)
        
        sizes = x.shape
        x = x.reshape(sizes[0], sizes[1], sizes[2]* sizes[3])
        # # print('x.reshape', x.shape)
        x = self.last_layer(x)
        # # x = x.reshape(sizes[0],16, self.anchor_dim)
        # x[:, :, self.num_y_steps:self.anchor_dim] = \
                # torch.sigmoid(x[:, :, self.num_y_steps:self.anchor_dim])

        # x = self.up1(x, x1)
        # print('bev up 1', x.shape)
        # x = self.up2(x)
        # print('bev up2', x.shape)

        return x



#  Lane Prediction Head: through a series of convolutions with no padding in the y dimension, the feature maps are
#  reduced in height, and finally the prediction layer size is N × 1 × 3 ·(3 · K + 1)
class LanePredictionHead(nn.Module):
    def __init__(self, num_lane_type, num_y_steps, batch_norm=False):
        super(LanePredictionHead, self).__init__()
        self.num_lane_type = num_lane_type
        self.num_y_steps = num_y_steps
        self.anchor_dim = 3*self.num_y_steps + 1
        layers = []
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)

        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        # input to layer--> [8, 64, 26, 16]
        # output after layer [8, 64, 4, 16]
        self.features = nn.Sequential(*layers)

        # x suppose to be N X 64 X 4 X ipm_w/8, need to be reshaped to N X 256 X ipm_w/8 X 1
        # TODO: use large kernel_size in x or fc layer to estimate z with global parallelism
        dim_rt_layers = []
        dim_rt_layers += make_one_layer(256, 128, kernel_size=(5, 1), padding=(2, 0), batch_norm=batch_norm)
        dim_rt_layers += [nn.Conv2d(128, self.num_lane_type*self.anchor_dim, kernel_size=(5, 1), padding=(2, 0))]
        # Output after the layer [8, 93, 16, 1], each anchor line has 3 types, each anchor line has 31 (x,z,vis + 1 ex_pro)values
        self.dim_rt = nn.Sequential(*dim_rt_layers)

    def forward(self, x):
        print('x before features', x.shape)
        x = self.features(x)
        print('x after features', x.shape)
        # x suppose to be N X 64 X 4 X ipm_w/8, reshape to N X 256 X ipm_w/8 X 1
        sizes = x.shape
        x = x.reshape(sizes[0], sizes[1]*sizes[2], sizes[3], 1)
        x = self.dim_rt(x)
        print('x after dim_rt', x.shape)
        x = x.squeeze(-1).transpose(1, 2)
        # apply sigmoid to the probability terms to make it in (0, 1)
        for i in range(self.num_lane_type):
            x[:, :, i*self.anchor_dim + 2*self.num_y_steps:(i+1)*self.anchor_dim] = \
                torch.sigmoid(x[:, :, i*self.anchor_dim + 2*self.num_y_steps:(i+1)*self.anchor_dim])
        return x


class Lane3DPredictionHead(nn.Module):
    def __init__(self, num_lane_type, num_y_steps,fmap_mapping_interp_index,
                                                  fmap_mapping_interp_weight, batch_norm=False):
        super(Lane3DPredictionHead, self).__init__()
        self.num_lane_type = num_lane_type
        self.num_y_steps = num_y_steps
        self.anchor_dim = 3*self.num_y_steps + 1
        layers = []
        # k=3 with pad=0 -> decreases the size by 2
        layers += make_one_3d_layer(64, 64, kernel_size=3, padding=(0, 0, 1), batch_norm=batch_norm)
        layers += make_one_3d_layer(64, 64, kernel_size=3, padding=(0, 0, 1), batch_norm=batch_norm)
        self.features_3d = nn.Sequential(*layers)
        
        layers = []
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)

        # k=5 with pad=0 -> decreases the size by 4
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        # input to layer--> [8, 64, 26, 16]
        # output after layer [8, 64, 4, 16]
        self.features = nn.Sequential(*layers)

        # x suppose to be N X 64 X 4 X ipm_w/8, need to be reshaped to N X 256 X ipm_w/8 X 1
        # TODO: use large kernel_size in x or fc layer to estimate z with global parallelism
        dim_rt_layers = []
        dim_rt_layers += make_one_layer(256, 128, kernel_size=(5, 1), padding=(2, 0), batch_norm=batch_norm)
        dim_rt_layers += [nn.Conv2d(128, self.num_lane_type*self.anchor_dim, kernel_size=(5, 1), padding=(2, 0))]
        # Output after the layer [8, 93, 16, 1], each anchor line has 3 types, each anchor line has 31 (x,z,vis + 1 ex_pro)values
        self.dim_rt = nn.Sequential(*dim_rt_layers)

        self.use_default_anchor = True
        if fmap_mapping_interp_index is not None and fmap_mapping_interp_weight is not None:
            self.use_default_anchor = False
            self.fmap_mapping_interp_index = torch.tensor(fmap_mapping_interp_index)
            self.fmap_mapping_interp_weight = torch.tensor(fmap_mapping_interp_weight)
            # if not no_cuda:
            #     self.fmap_mapping_interp_index = self.fmap_mapping_interp_index.cuda()
            #     self.fmap_mapping_interp_weight = self.fmap_mapping_interp_weight.cuda()


    def forward(self, x):

        # print('x before features3d', x.shape)
        x = self.features_3d(x)
        sizes = x.shape
        x = x.reshape(sizes[0], sizes[1]*sizes[2], sizes[3], sizes[4])
        # print('x after features3d', x.shape)

        if not self.use_default_anchor:
            # multi-gpu setting
            batch_size, channel, fmap_h, fmap_w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            sheared_feature_map = torch.zeros((batch_size, channel, fmap_h, fmap_w*6)).to(x.device)
            v_arange = torch.arange(fmap_h).unsqueeze(dim=1).repeat(1,fmap_w*6).type(torch.int64).to(x.device)
            self.fmap_mapping_interp_index = self.fmap_mapping_interp_index.type(torch.int64).to(x.device)
            self.fmap_mapping_interp_weight = self.fmap_mapping_interp_weight.type(torch.int64).to(x.device)
            # print('v_arange',v_arange.shape)
            # print('self.fmap_mapping_interp_index',self.fmap_mapping_interp_index.shape)
            # print('self.fmap_mapping_interp_weight',self.fmap_mapping_interp_weight.shape)
            # print('x',x.shape)
            for batch_idx, x_feature_map in enumerate(x):
                # if True:
                # print("v_arange device: " + str(v_arange.device))
                # print("self.fmap_mapping_interp_index device: " + str(self.fmap_mapping_interp_index.device))
                # print("self.fmap_mapping_interp_weight device: " + str(self.fmap_mapping_interp_weight.device))
                # print("sheared_feature_map device: " + str(sheared_feature_map.device))
                # print("batch_idx device: " + str(v_arange.device))
                # print("x_feature_map device: " + str(x_feature_map.device))

                sheared_feature_map[batch_idx] = \
                    x_feature_map[:, v_arange, self.fmap_mapping_interp_index[:,:,0]] * self.fmap_mapping_interp_weight[:,:,0] + \
                    x_feature_map[:, v_arange, self.fmap_mapping_interp_index[:,:,1]] * self.fmap_mapping_interp_weight[:,:,1]
            x = torch.cat((x, sheared_feature_map), dim=3)
        
        # print('x before features', x.shape)
        x = self.features(x)
        # print('x after features', x.shape)
        # x suppose to be N X 64 X 4 X ipm_w/8, reshape to N X 256 X ipm_w/8 X 1
        sizes = x.shape
        x = x.reshape(sizes[0], sizes[1]*sizes[2], sizes[3], 1)
        # print('x after features reshape', x.shape)
        x = self.dim_rt(x)
        # print('x after dim_rt', x.shape)
        x = x.squeeze(-1).transpose(1, 2)
        # apply sigmoid to the probability terms to make it in (0, 1)
        for i in range(self.num_lane_type):
            x[:, :, i*self.anchor_dim + 2*self.num_y_steps:(i+1)*self.anchor_dim] = \
                torch.sigmoid(x[:, :, i*self.anchor_dim + 2*self.num_y_steps:(i+1)*self.anchor_dim])
        return x



class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC, num_y_steps, intrins,args):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        
        
        self.num_y_steps = num_y_steps 
        self.num_lane_type = 3
        
        # print("self.grid_conf['xbound']", self.grid_conf['xbound'])
        # print("self.grid_conf['ybound']", self.grid_conf['ybound'])

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                  self.grid_conf['ybound'],
                                  self.grid_conf['zbound'],
                                  )
                                              
        # print("nx", nx)
        
        self.dx_ = nn.Parameter(dx, requires_grad=False)
        self.bx_ = nn.Parameter(bx, requires_grad=False)
        self.nx_ = nn.Parameter(nx, requires_grad=False)      

        self.downsample = 16
        self.camC = 64
        self.frustum_ = self.create_frustum().cuda(0)
        self.D, _, _, _ = self.frustum_.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        # self.bevencode = BevEncode(inC=self.camC, outC=outC, num_y_steps = self.num_y_steps)

        # self.encoder = make_layers([64, 'M', 64, 'M', 64, 'M', 64,'M', 64,'M', 64,], self.camC, batch_norm=True)
        # Conv layers to convert original resolution binary map to target resolution with high-dimension
        # self.encoder = make_layers([64, 'M', 64, 'M', 64, 'M', 64], self.camC, batch_norm=True)
        self.encoder = make_3d_layers([64, 'M', 64, 'M', 64, 'M', 64], self.camC, batch_norm=True)
        
        # self.lane_out = LanePredictionHead(32, self.num_lane_type, self.num_y_steps, batch_norm=True)
        # self.lane_out = LanePredictionHead(self.num_lane_type, self.num_y_steps, True)
        self.lane_out = Lane3DPredictionHead(self.num_lane_type, self.num_y_steps,
                                                  args.fmap_mapping_interp_index,
                                                  args.fmap_mapping_interp_weight,True)
        
        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
        
        
        self.w_ce = nn.Parameter(torch.tensor([1.0]))###>
        self.w_l1 = nn.Parameter(torch.tensor([1.0]))###>
        
        ################################
    
        self.intrins = nn.Parameter(torch.from_numpy(intrins), requires_grad=False)
        
        ################################
        effNetB0_dimList = [16, 24, 40, 112, 1280]
        self.neck = nn.Sequential(*make_one_layer(effNetB0_dimList[0], args.feature_channels, batch_norm=True),
                                  *make_one_layer(args.feature_channels, args.feature_channels, batch_norm=True))
        # 2d lane detector
        self.shared_encoder = FrontViewPathway(args.feature_channels, args.num_proj)
        stride = 2
        self.laneatt_head = LaneATTHead(stride * pow(2, args.num_proj - 1),
                                               args.feature_channels * pow(2, args.num_proj - 2), # no change in last proj
                                               args.im_anchor_origins,
                                               args.im_anchor_angles,
                                               img_w=args.resize_w,
                                               img_h=args.resize_h,
                                               S=args.S,
                                               anchor_feat_channels=args.anchor_feat_channels,
                                               num_category=1)
        # segmentation head
        self.segment_head = SegmentHead(channels=args.feature_channels)
        # uncertainty loss weight
        self.uncertainty_loss = nn.Parameter(torch.tensor([args._3d_vis_loss_weight,
                                                            args._3d_prob_loss_weight,
                                                            args._3d_reg_loss_weight,
                                                            args._2d_vis_loss_weight,
                                                            args._2d_prob_loss_weight,
                                                            args._2d_reg_loss_weight,
                                                            args._seg_loss_weight]), requires_grad=True)
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']# after image resize
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)#repeat the vector size fW, fH times then repeat the matrix D times
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        #ex frustum[0] would have xs=[1,2,3...], ys=[1,1,1,..], d=[4,4,4..]
        #ex frustum[1] would have xs=[1,2,3...], ys=[2,2,2,..], d=[4,4,4..]
        frustum = torch.stack((xs, ys, ds), -1) #each voxel or a cell is x,y, depth #as if it scans the space #stack over the highest values axis
        return frustum#nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3 #B ->batch size, N -># of cams
        """
        B, N, _ = trans.shape

        #seems if the frustum is the positions of the pixels in the image after augmenation so it has to be reversed
        # undo post-transformation
        #I believe he removes the augmentatin
        # B x N x D x H x W x 3
        points = self.frustum_ - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        #x,y coordinates are multiplied by z
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape
        # print("x before encode", x.shape)
        x = x.view(B*N, C, imH, imW)
        x, encoder_feat = self.camencode(x)
        # print("x after encode", x.shape)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x, encoder_feat

    def voxel_pooling(self, geom_feats, x):
    
        B, N, D, H, W, C = x.shape
        # print('x.shape', x.shape)
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        #bound the geom_feats in the bev projected area
        geom_feats = ((geom_feats - (self.bx_ - self.dx_/2.)) / self.dx_).long()
        # print('geom_feats.shape', geom_feats.shape)
        geom_feats = geom_feats.view(Nprime, 3)
        #batch_ix size (Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # print('x before kept', x.shape)
        # filter out points that are outside box
        #seems nx is number of cells in the grid in x,y,z direction
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx_[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx_[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx_[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        # print('x after kept', x.shape)

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx_[1] * self.nx_[2] * B)\
            + geom_feats[:, 1] * (self.nx_[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # print("before cum sum", x.shape)
        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # print("after cum sum", x.shape)
        # griddify (B x C x Z x X x Y)
        # print("self.nx_->voxel_pooling", self.nx_)
        # print("x shape", x.shape)
        # print("geom_feats shape", geom_feats.shape)
        # final = torch.zeros((B, C, self.nx_[2], self.nx_[0], self.nx_[1]), device=x.device)
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        
        final = torch.zeros((B, C, self.nx_[2], self.nx_[1], self.nx_[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
            
        # print("final dim before collapse", final.shape)
        
        # collapse Z
        # final2d = torch.cat(final.unbind(dim=2), 1)# seems dim 1 is C*Z
        # print("final dim after collapse", final.shape)

        return final#,final2d

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        #I believe that geom is a frustum describing each pixel in the input images in the 3d world!
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x, encoder_feat = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x, encoder_feat

    def forward(self, x, rot_cam2ego, trans_cam2ego, post_rots, post_trans):
    
        # print("self.nx_->forward", self.nx_)
        x, encoder_feat = self.get_voxels(x, rot_cam2ego, trans_cam2ego, self.intrins, post_rots, post_trans)

##############################
        
        out_featList = encoder_feat#self.encoder(input)
        # print('out_featList', out_featList.shape)
        neck_out = self.neck(out_featList)
        # print('neck_out', neck_out.shape)
        frontview_features = self.shared_encoder(neck_out)
        '''
            frontview_features_0 size: torch.Size([4, 128, 180, 240])
            frontview_features_1 size: torch.Size([4, 256, 90, 120])
            frontview_features_2 size: torch.Size([4, 512, 45, 60])
            frontview_features_3 size: torch.Size([4, 512, 22, 30])
        '''
        frontview_final_feat = frontview_features[-1]
        # print('frontview_final_feat', frontview_final_feat.shape)

        laneatt_proposals_list = self.laneatt_head(frontview_final_feat)

        # print('x before segmentation', x.shape)
        # x2d = self.bevencode(x2d)
        x2d = torch.sum(x,dim=2)
        # print('x2d', x2d.shape)
        pred_seg_bev_map = self.segment_head(x2d)
        # print('pred_seg_bev_map', pred_seg_bev_map.shape)

        # seperate loss weight
        uncertainty_loss = torch.tensor(1.0).to(x.device) * self.uncertainty_loss.to(x.device)

##############################

        # print('********************* x size before lane pridection')
        # print(x.shape)
        # x = self.bevencode(x)
        x = self.encoder(x)
        # print('x encoder', x.shape)
        # convert top-view features to anchor output
        x = self.lane_out(x)
        # print('final out shape')
        # print(out.shape)
        return laneatt_proposals_list, pred_seg_bev_map, x, self.w_l1, self.w_ce,uncertainty_loss


def compile_model(grid_conf, data_aug_conf, outC, num_y_steps, intrins,args):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC, num_y_steps, intrins, args)
