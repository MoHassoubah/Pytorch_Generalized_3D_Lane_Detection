"""
Loss functions

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import numpy as np
import torch
import torch.nn as nn


class Laneline_loss_3D(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is based on real 3D X, Y, Z.

    loss = loss1 + loss2 + loss2
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    loss3: error in estimating pitch and camera heights
    """
    def __init__(self, num_types, anchor_dim, pred_cam):
        super(Laneline_loss_3D, self).__init__()
        self.num_types = num_types
        self.anchor_dim = anchor_dim
        self.pred_cam = pred_cam

    def forward(self, pred_3D_lanes, gt_3D_lanes, pred_hcam, gt_hcam, pred_pitch, gt_pitch):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (2K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor value N x ipm_w/8 x 3 x 2K
        pred_class = pred_3D_lanes[:, :, :, -1].unsqueeze(-1)
        pred_anchors = pred_3D_lanes[:, :, :, :-1]
        gt_class = gt_3D_lanes[:, :, :, -1].unsqueeze(-1)
        gt_anchors = gt_3D_lanes[:, :, :, :-1]

        loss1 = -torch.sum(gt_class*torch.log(pred_class + torch.tensor(1e-9)) +
                           (torch.ones_like(gt_class)-gt_class) *
                           torch.log(torch.ones_like(pred_class)-pred_class + torch.tensor(1e-9)))
        # applying L1 norm does not need to separate X and Z
        loss2 = torch.sum(torch.norm(gt_class*(pred_anchors-gt_anchors), p=1, dim=3))
        if not self.pred_cam:
            return loss1+loss2
        loss3 = torch.sum(torch.abs(gt_pitch-pred_pitch))+torch.sum(torch.abs(gt_hcam-pred_hcam))
        return loss1+loss2+loss3


class Laneline_loss_gflat(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is in flat ground space X', Y' and real 3D Z. Visibility estimation is also included.

    loss = loss0 + loss1 + loss2 + loss2
    loss0: cross entropy loss for lane point visibility
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    loss3: error in estimating pitch and camera heights
    """
    def __init__(self, num_types, num_y_steps, pred_cam):
        super(Laneline_loss_gflat, self).__init__()
        self.num_types = num_types
        self.num_y_steps = num_y_steps
        self.anchor_dim = 3*self.num_y_steps + 1
        self.pred_cam = pred_cam

    def forward(self, pred_3D_lanes, gt_3D_lanes, pred_hcam, gt_hcam, pred_pitch, gt_pitch):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (3K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor value N x ipm_w/8 x 3 x 2K
        pred_class = pred_3D_lanes[:, :, :, -1].unsqueeze(-1)
        pred_anchors = pred_3D_lanes[:, :, :, :2*self.num_y_steps]
        pred_visibility = pred_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]
        gt_class = gt_3D_lanes[:, :, :, -1].unsqueeze(-1)
        gt_anchors = gt_3D_lanes[:, :, :, :2*self.num_y_steps]
        gt_visibility = gt_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]

        # cross-entropy loss for visibility
        loss0 = -torch.sum(
            gt_visibility*torch.log(pred_visibility + torch.tensor(1e-9)) +
            (torch.ones_like(gt_visibility) - gt_visibility + torch.tensor(1e-9)) *
            torch.log(torch.ones_like(pred_visibility) - pred_visibility + torch.tensor(1e-9)))/self.num_y_steps
        # cross-entropy loss for lane probability
        loss1 = -torch.sum(
            gt_class*torch.log(pred_class + torch.tensor(1e-9)) +
            (torch.ones_like(gt_class)-gt_class) *
            torch.log(torch.ones_like(pred_class) - pred_class + torch.tensor(1e-9)))
        # applying L1 norm does not need to separate X and Z
        loss2 = torch.sum(torch.norm(gt_class*torch.cat((gt_visibility, gt_visibility), 3) *
                                     (pred_anchors-gt_anchors), p=1, dim=3))
        if not self.pred_cam:
            return loss0+loss1+loss2
        loss3 = torch.sum(torch.abs(gt_pitch-pred_pitch))+torch.sum(torch.abs(gt_hcam-pred_hcam))
        return loss0+loss1+loss2+loss3


class Laneline_loss_gflat_3D(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is in flat ground space X', Y' and real 3D Z. Visibility estimation is also included.
    The X' Y' and Z estimation will be transformed to real X, Y to compare with ground truth. An additional loss in
    X, Y space is expected to guide the learning of features to satisfy the geometry constraints between two spaces

    loss = loss0 + loss1 + loss2 + loss2
    loss0: cross entropy loss for lane point visibility
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    loss3: error in estimating pitch and camera heights
    """
    def __init__(self, batch_size,anchor_num, num_types, anchor_x_steps, anchor_y_steps, x_off_std, y_off_std, z_std, pred_cam=False, no_cuda=False):
        super(Laneline_loss_gflat_3D, self).__init__()
        self.batch_size = batch_size
        self.num_types = num_types
        print('x_off_std',x_off_std.shape)
        self.num_x_steps = anchor_x_steps.shape[0]
        self.num_y_steps = anchor_y_steps.shape[0]
        self.anchor_dim = 3*self.num_y_steps + 1
        self.pred_cam = pred_cam
        self.anchor_num = anchor_num

        # prepare broadcast anchor_x_tensor, anchor_y_tensor, std_X, std_Y, std_Z
        tmp_zeros = torch.zeros(self.batch_size, self.anchor_num, self.num_types, self.num_y_steps)
        self.x_off_std = torch.tensor(x_off_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.y_off_std = torch.tensor(y_off_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.z_std = torch.tensor(z_std.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.anchor_x_tensor = torch.tensor(anchor_x_steps.astype(np.float32)).reshape(1, self.anchor_num, 1, self.num_y_steps) + tmp_zeros
        self.anchor_y_tensor = torch.tensor(anchor_y_steps.astype(np.float32)).reshape(1, 1, 1, self.num_y_steps) + tmp_zeros
        self.anchor_x_tensor = self.anchor_x_tensor/self.x_off_std
        self.anchor_y_tensor = self.anchor_y_tensor/self.y_off_std

        if not no_cuda:
            self.x_off_std = self.x_off_std.cuda()
            self.y_off_std = self.y_off_std.cuda()
            self.z_std = self.z_std.cuda()
            self.anchor_x_tensor = self.anchor_x_tensor.cuda()
            self.anchor_y_tensor = self.anchor_y_tensor.cuda()

    def calParrallelismLoss(self, pred_Xoff, pred_Yoff, pred_Z, gt_class, gt_visibility):


        parrallelism_loss = torch.tensor(0)
        un_normPred_X = (pred_Xoff + self.anchor_x_tensor)*self.x_off_std
        un_normPred_Y = (pred_Yoff + self.anchor_y_tensor)*self.y_off_std
        un_normPred_Z = pred_Z*self.z_std
        for typ in range(3):
            # gt_class[:,:,0] is batch_size x num_anchors x 1
            idx_non_zero_anchors = torch.nonzero(gt_class[:,:,typ]) #t
            # ######print('idx_non_zero_anchors', idx_non_zero_anchors.shape)
            _, counts = torch.unique_consecutive(idx_non_zero_anchors[:,0], return_counts=True)
            idx_imgs_val_lanes_greater_than_2 = torch.nonzero(counts>=2)#z
            
            # x_values = pred_Xoff[:,:,typ,:] + self.anchor_x_tensor[:,:,typ,:]
            # x_values = x_values * self.x_off_std[:,:,typ,:]
            x_values = un_normPred_X[:,:,typ,:]
            # y_values = pred_Yoff[:,:,typ,:] + self.anchor_y_tensor[:,:,typ,:]
            # y_values = y_values * self.y_off_std[:,:,typ,:]
            y_values = un_normPred_Y[:,:,typ,:]
            # z_valus = pred_Z[:,:,typ,:] * self.z_std[:,:,typ,:]
            z_valus = un_normPred_Z[:,:,typ,:]
            num_dims = len(x_values.shape)
            # batch_size x num_anchors x num_point (3 y steps) x 3 (x,y,z)
            x_y_z_vect = torch.cat((torch.cat((x_values.unsqueeze(num_dims),y_values.unsqueeze(num_dims))
            ,dim=num_dims),  z_valus.unsqueeze(num_dims)),dim=num_dims)
            # ######print('x_y_z_vect=', x_y_z_vect.shape)

            # batch_size x num_anchors x num_line_vect (num_point-1)
            typ_gt_visibility = gt_visibility[:,:,typ,:]
            # img_gt_visibility = img_gt_visibility[idx_imgs_val_lanes_greater_than_2].squeeze(1)

            for z_i in idx_imgs_val_lanes_greater_than_2:
                val_anch_idx = idx_non_zero_anchors[:,1][torch.nonzero(idx_non_zero_anchors[:,0]==z_i)]
                
                # ######print('val_anch_idx=', val_anch_idx.shape)
                # get the line vectors of each lane anchor
                # num_anchors x num_line_vect (num_point-1) x 3 (x,y,z)
                img_x_y_z_vect = x_y_z_vect[z_i,val_anch_idx,:,:].squeeze(1)
                # ######print('x_y_z_vect consider the current image=', img_x_y_z_vect.shape)
                # num_valid_anchors x num points
                img_gt_visibility = typ_gt_visibility[z_i,val_anch_idx,:].squeeze(1)
                # ######print('img_gt_visibility consider the current imge=', img_gt_visibility.shape)

                lane_line_vect = img_x_y_z_vect[:,1:self.num_y_steps,:] - \
                        img_x_y_z_vect[:,0:self.num_y_steps-1,:]
                # ######print('lane_line_vect=', lane_line_vect.shape)
                        
                lane_line_vect = lane_line_vect / (torch.norm(lane_line_vect,dim=-1).unsqueeze(2) + 1e-07)
                
                valid_line_vect = img_gt_visibility[:,0:-1]*img_gt_visibility[:,1:]
                # ######print('vali_line_vect sum 1', valid_line_vect.sum())
                # ######print('valid_linde_vect=', valid_line_vect.shape)

                # ######print('valid_line_vect before ', valid_line_vect)

                valid_line_vect = valid_line_vect[1:,:]*valid_line_vect[0:-1,:]
                
                # ######print('valid_line_vect sum 2', valid_line_vect.sum())
                # ######print('valid_line_vect after neigh anchor op=', valid_line_vect.shape)

                #dot product of each anchor line vector with the neigbouring anchor next to it
                # num_anchors-1 x num_line_vect (dot product value for each vect segment)
                # each one of num_anchors-1 x num_line_vect is a dot product between matching 
                # segements at y-coords pairs
                neighbor_anchor_dot_prod = lane_line_vect[1:,:]*lane_line_vect[0:-1,:]
                neighbor_anchor_dot_prod = neighbor_anchor_dot_prod.sum(dim=-1)
                # ######print('neighbor_anchor_dot_prod', neighbor_anchor_dot_prod.shape)

                parrallelism_out = torch.ones_like(neighbor_anchor_dot_prod) - neighbor_anchor_dot_prod
                # ######print('parrallelism_out sum 1', parrallelism_out.sum())

                parrallelism_out = parrallelism_out * valid_line_vect
                # ######('parrallelism_out sum 2', parrallelism_out.sum())

                # batch_size x num_anchors-1 x 1
                # valid_anchors = gt_class[:,1:,0]*gt_class[:,0:-1,0]
                # print('gt_class sum', gt_class[:,:,0].sum())
                # ######print('valid_anchors sum', valid_anchors.sum())
                # ######print('valid_anchors', valid_anchors.shape)

                # batch_size x num_anchors-1 x num_line_vect (dot product value for each vect segment)
                # parrallelism_out = parrallelism_out * valid_anchors
                # ######print('parrallelism_out sum 3', parrallelism_out.sum())
                # the norm measure for every anchor the degree of parallelism

                parrallelism_loss =  parrallelism_loss + \
                    torch.sum( torch.norm(parrallelism_out, p=1, dim=-1))
                # ######print('parrallelism_loss=', parrallelism_loss)
        # print('return parrallelism_loss=', parrallelism_loss)
        return parrallelism_loss

    def forward(self, pred_3D_lanes, gt_3D_lanes, pred_hcam, gt_hcam, pred_pitch, gt_pitch):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (3K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor values N x ipm_w/8 x 3 x 2K, visibility N x ipm_w/8 x 3 x K
        pred_class = pred_3D_lanes[:, :, :, -1].unsqueeze(-1)
        pred_anchors = pred_3D_lanes[:, :, :, :2*self.num_y_steps]
        pred_visibility = pred_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]
        gt_class = gt_3D_lanes[:, :, :, -1].unsqueeze(-1)
        gt_anchors = gt_3D_lanes[:, :, :, :2*self.num_y_steps]
        gt_visibility = gt_3D_lanes[:, :, :, 2*self.num_y_steps:3*self.num_y_steps]

        # cross-entropy loss for visibility
        loss0 = -torch.sum(
            gt_visibility*torch.log(pred_visibility + torch.tensor(1e-9)) +
            (torch.ones_like(gt_visibility) - gt_visibility + torch.tensor(1e-9)) *
            torch.log(torch.ones_like(pred_visibility) - pred_visibility + torch.tensor(1e-9)))/self.num_y_steps
        # cross-entropy loss for lane probability
        loss1 = -torch.sum(
            gt_class*torch.log(pred_class + torch.tensor(1e-9)) +
            (torch.ones_like(gt_class) - gt_class) *
            torch.log(torch.ones_like(pred_class) - pred_class + torch.tensor(1e-9)))
        # applying L1 norm does not need to separate X and Z
        loss2 = torch.sum(
            torch.norm(gt_class*torch.cat((gt_visibility, gt_visibility), 3)*(pred_anchors-gt_anchors), p=1, dim=3))

        # compute loss in real 3D X, Y space, the transformation considers offset to anchor and normalization by std
        pred_Xoff_g = pred_anchors[:, :, :, :self.num_y_steps]
        pred_Z = pred_anchors[:, :, :, self.num_y_steps:2*self.num_y_steps]
        gt_Xoff_g = gt_anchors[:, :, :, :self.num_y_steps]
        gt_Z = gt_anchors[:, :, :, self.num_y_steps:2*self.num_y_steps]
        pred_hcam = pred_hcam.reshape(self.batch_size, 1, 1, 1)
        gt_hcam = gt_hcam.reshape(self.batch_size, 1, 1, 1)

        pred_Xoff = (1 - pred_Z * self.z_std / pred_hcam) * pred_Xoff_g - pred_Z * self.z_std / pred_hcam * self.anchor_x_tensor
        pred_Yoff = -pred_Z * self.z_std / pred_hcam * self.anchor_y_tensor
        gt_Xoff = (1 - gt_Z * self.z_std / gt_hcam) * gt_Xoff_g - gt_Z * self.z_std / gt_hcam * self.anchor_x_tensor
        gt_Yoff = -gt_Z * self.z_std / gt_hcam * self.anchor_y_tensor
        # loss3 = torch.sum(
        #     torch.norm(
        #         gt_class * torch.cat((gt_visibility, gt_visibility), 3) *
        #         (torch.cat((pred_Xoff, pred_Yoff), 3) - torch.cat((gt_Xoff, gt_Yoff), 3)), p=1, dim=3))

        # loss5 = self.calParrallelismLoss(pred_Xoff, pred_Yoff, pred_Z, gt_class, gt_visibility)

        # if not self.pred_cam:
        return loss0+loss1+loss2 , {'vis_loss': loss0, 'prob_loss': loss1, 'reg_loss': loss2}#, 'cam_pred_loss': loss3} #+loss3
        # loss4 = torch.sum(torch.abs(gt_pitch-pred_pitch)) + torch.sum(torch.abs(gt_hcam-pred_hcam))
        # return loss0+loss1+loss3+loss4+loss5 #+loss2


# unit test
if __name__ == '__main__':
    num_types = 3

    # for Laneline_loss_3D
    print('Test Laneline_loss_3D')
    anchor_dim = 2*6 + 1
    pred_cam = True
    criterion = Laneline_loss_3D(num_types, anchor_dim, pred_cam)
    criterion = criterion.cuda()

    pred_3D_lanes = torch.rand(8, 26, num_types*anchor_dim).cuda()
    gt_3D_lanes = torch.rand(8, 26, num_types*anchor_dim).cuda()
    pred_pitch = torch.ones(8).float().cuda()
    gt_pitch = torch.ones(8).float().cuda()
    pred_hcam = torch.ones(8).float().cuda()
    gt_hcam = torch.ones(8).float().cuda()

    loss = criterion(pred_3D_lanes, gt_3D_lanes, pred_pitch, gt_pitch, pred_hcam, gt_hcam)
    print(loss)

    # for Laneline_loss_gflat
    print('Test Laneline_loss_gflat')
    num_y_steps = 6
    anchor_dim = 3*num_y_steps + 1
    pred_cam = True
    criterion = Laneline_loss_gflat(num_types, num_y_steps, pred_cam)
    criterion = criterion.cuda()

    pred_3D_lanes = torch.rand(8, 26, num_types*anchor_dim).cuda()
    gt_3D_lanes = torch.rand(8, 26, num_types*anchor_dim).cuda()
    pred_pitch = torch.ones(8).float().cuda()
    gt_pitch = torch.ones(8).float().cuda()
    pred_hcam = torch.ones(8).float().cuda()
    gt_hcam = torch.ones(8).float().cuda()

    loss = criterion(pred_3D_lanes, gt_3D_lanes, pred_pitch, gt_pitch, pred_hcam, gt_hcam)

    print(loss)

    # for Laneline_loss_gflat_3D
    print('Test Laneline_loss_gflat_3D')
    batch_size = 8
    anchor_x_steps = np.linspace(-10, 10, 26, endpoint=True)
    anchor_y_steps = np.array([3, 5, 10, 20, 30, 40, 50, 60, 80, 100])
    num_y_steps = anchor_y_steps.shape[0]
    x_off_std = np.ones(num_y_steps)
    y_off_std = np.ones(num_y_steps)
    z_std = np.ones(num_y_steps)
    pred_cam = True
    criterion = Laneline_loss_gflat_3D(batch_size, num_types, anchor_x_steps, anchor_y_steps, x_off_std, y_off_std, z_std, pred_cam, no_cuda=False)
    # criterion = criterion.cuda()

    anchor_dim = 3*num_y_steps + 1
    pred_3D_lanes = torch.rand(batch_size, 26, num_types*anchor_dim).cuda()
    gt_3D_lanes = torch.rand(batch_size, 26, num_types*anchor_dim).cuda()
    pred_pitch = torch.ones(batch_size).float().cuda()
    gt_pitch = torch.ones(batch_size).float().cuda()
    pred_hcam = torch.ones(batch_size).float().cuda()*1.5
    gt_hcam = torch.ones(batch_size).float().cuda()*1.5

    loss = criterion(pred_3D_lanes, gt_3D_lanes, pred_pitch, gt_pitch, pred_hcam, gt_hcam)

    print(loss)