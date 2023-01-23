"""
Batch test code for Gen-LaneNet with new anchor extension. It predicts 3D lanes per image.

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import numpy as np
import torch
import torch.optim
import glob
from tqdm import tqdm
from dataloader.Load_Data_3DLane_ext import *
from networks import GeoNet3D_ext, erfnet
from tools.utils import *
from tools import eval_3D_lane
from networks.models import compile_model


def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    ckpt_name = []
    cnt = 0
    for name, param in state_dict.items():
        if name[7:] not in list(own_state.keys()) or 'output_conv' in name:
            ckpt_name.append(name)
            # continue
        own_state[name[7:]].copy_(param)
        cnt += 1
    print('#reused param: {}'.format(cnt))
    return model


def deploy(args, loader, dataset, model_seg, model_geo, vs_saver, test_gt_file, vis=False, epoch=0):

    # model deploy mode
    model_geo.eval()

    # read ground-truth lanes for later evaluation
    test_set_labels = [json.loads(line) for line in open(test_gt_file).readlines()]

    # Only forward pass, hence no gradients needed
    with torch.no_grad():
        with open(lane_pred_file, 'w') as jsonFile:
            # Start validation loop
            for i, (input, _, gt, idx, gt_hcam, gt_pitch,rots, trans, post_rot, post_tran) in tqdm(enumerate(loader)):
                if not args.no_cuda:
                    input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
                    gt_hcam = gt_hcam.cuda(0)
                    gt_pitch = gt_pitch.cuda(0)
                    rots = rots.cuda(0)
                    trans = trans.cuda(0)
                    post_rot = post_rot.cuda(0)
                    post_tran = post_tran.cuda(0)
                input = input.contiguous()
                input = torch.autograd.Variable(input)

                # # if not args.fix_cam and not args.pred_cam:
                # # ATTENTION: here requires to update with test dataset args
                # model_geo.update_projection(args, gt_hcam, gt_pitch)

                # Evaluate model
                # try:
                    # output_seg = model_seg(input, no_lane_exist=True)
                    # # output1 = F.softmax(output1, dim=1)
                    # output_seg = output_seg.softmax(dim=1)
                    # output_seg = output_seg / torch.max(torch.max(output_seg, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
                    # output_seg = output_seg[:, 1:, :, :]
                pred_pitch = gt_pitch
                pred_hcam = gt_hcam
                output_geo, w_l1, w_ce = model_geo(input,
                    rots,
                    trans,
                    post_rot,
                    post_tran,
                    )
                # except RuntimeError as e:
                    # print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                    # print(e)
                    # continue

                input = input.squeeze(1)
                gt = gt.data.cpu().numpy()
                output_geo = output_geo.data.cpu().numpy()
                pred_pitch = pred_pitch.data.cpu().numpy().flatten()
                pred_hcam = pred_hcam.data.cpu().numpy().flatten()

                # unormalize lane outputs
                num_el = input.size(0)
                for j in range(num_el):
                    unormalize_lane_anchor(gt[j], dataset)
                    unormalize_lane_anchor(output_geo[j], dataset)

                if vis:
                    # Plot curves in two views
                    vs_saver.save_result_new(dataset, args.vis_folder, epoch, i, idx,
                                             input, gt, output_geo, pred_pitch, pred_hcam, evaluate=False)#args.evaluate)

                # visualize and write results
                for j in range(num_el):
                    im_id = idx[j]
                    H_g2im, P_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[j])
                    """
                        save results in test dataset format
                    """
                    json_line = test_set_labels[im_id]
                    lane_anchors = output_geo[j]
                    # convert to json output format
                    lanelines_pred, centerlines_pred, lanelines_prob, centerlines_prob =\
                        compute_3d_lanes_all_prob(lane_anchors, dataset.anchor_dim,
                                                  dataset.anchor_x_steps, args.anchor_y_steps, pred_hcam[j])
                    json_line["laneLines"] = lanelines_pred
                    json_line["centerLines"] = centerlines_pred
                    json_line["laneLines_prob"] = lanelines_prob
                    json_line["centerLines_prob"] = centerlines_prob
                    json.dump(json_line, jsonFile)
                    jsonFile.write('\n')

        # evaluation at varying thresholds
        # eval_stats_pr = evaluator.bench_one_submit_varying_probs(lane_pred_file, test_gt_file)
        # max_f_prob = eval_stats_pr['max_F_prob_th']

        # evaluate at the point with max F-measure. Additional eval of position error.
        eval_stats = evaluator.bench_one_submit(lane_pred_file, test_gt_file)#, prob_th=max_f_prob)

        # print("Metrics: AP, F-score, x error (Vclose), x error (close), x error (far), z error (Vclose), z error (close), z error (far)")
        print("Metrics: AP, F-score, x error (close), x error (far), z error (close), z error (far)")
        
        print(
            "Laneline:  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}".format(eval_stats[2], eval_stats[0],
                                                                         eval_stats[3], eval_stats[4],
                                                                         eval_stats[5], eval_stats[6]))
        print("Centerline:  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}".format(eval_stats[9], eval_stats[7],
                                                                             eval_stats[10], eval_stats[11],
                                                                             eval_stats[12], eval_stats[13]))
                                                                             
        # print(
        #     "Laneline:  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}".format(  eval_stats[2], eval_stats[0],
        #                                                                  eval_stats[3], eval_stats[4],
        #                                                                  eval_stats[5], eval_stats[6],
        #                                                                  eval_stats[7], eval_stats[8]))
        # print("Centerline:  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}  {:.3}".format( eval_stats[11], eval_stats[9],
        #                                                                      eval_stats[12], eval_stats[13],
        #                                                                      eval_stats[14], eval_stats[15],
        #                                                                      eval_stats[16], eval_stats[17]))

    return eval_stats


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = define_args()
    args = parser.parse_args()

    # manual settings
    args.dataset_dir = '../media/yuliangguo/DATA1/Datasets/Apollo_Sim_3D_Lane_Release/'  # raw data dir
    args.dataset_name = 'standard'#'illus_chg'  # choose a data split 'standard' / 'rare_subset' / 'illus_chg'
    args.mod = 'Gen_LaneNet_ext'  # model name
    test_name = 'test'  # test set name
    pretrained_feat_model = 'pretrained/erfnet_model_sim3d.tar'
    # vis = False  # choose to save visualization result
    vis = True  # choose to save visualization result
    # generate relative paths
    args.data_dir = ops.join('data_splits', args.dataset_name)
    args.save_path = os.path.join(ops.join('data_splits', args.dataset_name), args.mod)
    args.vis_folder = test_name + '_vis'
    if vis:
        mkdir_if_missing(os.path.join(args.save_path, 'example/' + args.vis_folder))
    test_gt_file = ops.join(args.data_dir, test_name + '.json')
    lane_pred_file = ops.join(args.save_path, test_name + '_pred_file.json')

    # load configuration for certain dataset
    sim3d_config(args)
    args.no_centerline = True
    args.y_ref = 5
    # define evaluator
    evaluator = eval_3D_lane.LaneEval(args)
    args.prob_th = 0.5

    # Check GPU availability
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn

    # Define network
    model_seg = erfnet.ERFNet(2)  # 2-class model
    # model_geo = GeoNet3D_ext.Net(args)
    
    #####################################
    #####################################
    
    ##############
    ##############
    lane_detection = {}
    
    lane_detection['y_ref'] = 5.0
    # # # lane_detection['top_view_region'] = np.array([[-10, 53], [10, 53], [-10, -47], [10, -47]])
    
    lane_detection['top_view_region'] = args.top_view_region #np.array([[-5, 53], [5, 53], [-5, 0], [5, 0]])
    
    # # # lane_detection['anchor_y_steps'] = np.array([-45, -40, -35, -30, -20, -10, 0, 10, 30, 50])
    # lane_detection['anchor_y_steps'] = np.array([0, 10, 30, 50])
    
    lane_detection['anchor_y_steps'] = args.anchor_y_steps #np.array([0,5, 10, 15,20,25,30,35,40, 50])
    
    lane_detection['num_y_steps'] = args.num_y_steps #len(lane_detection['anchor_y_steps'])
    # compute anchor steps
    x_min = lane_detection['top_view_region'][0, 0]
    x_max = lane_detection['top_view_region'][1, 0]
    lane_detection['x_min'] = x_min
    lane_detection['x_max'] = x_max
    lane_detection['y_min'] = lane_detection['top_view_region'][2, 1]
    lane_detection['y_max'] = lane_detection['top_view_region'][1, 1]
    lane_detection['ipm_w'] = 128
    lane_detection['n_anchors'] = int(args.ipm_w / 8)
    lane_detection['anchor_x_steps'] = np.linspace(x_min, x_max, lane_detection['n_anchors'], endpoint=True) #np.array([0])# #np.int(lane_detection['ipm_w']/8)
    # lane_detection['num_y_steps'] = len(lane_detection['anchor_y_steps'])
    # lane_detection['anchor_dim'] = lane_detection['num_y_steps']+1
    
    lane_detection['ref_id'] = np.argmin(np.abs(lane_detection['anchor_y_steps'] - lane_detection['y_ref'] ))
    
    lane_detection['line_names'] = ['road_divider', 'lane_divider']
    
    ##############
    ##############
    
    
    resolution_ratio = [0.25]#[0.3,0.25,0.22,0.15,0.1]
    H=1080
    W=1920
    
    for ratio in resolution_ratio:
    
    
        args.resize_w = int(W * ratio)
        args.resize_h = int(H * ratio)
        
        resize_lim=(0.193, 0.225)
        final_dim= (args.resize_h, args.resize_w) #(128, 352)
        bot_pct_lim=(0.0, 0.22)
        rot_lim=(-5.4, 5.4)
        rand_flip=True
        ncams=1#6
        max_grad_norm=5.0
        pos_weight=2.13
        logdir='./runs'
        
        data_aug_conf = {
                        # 'resize_lim': resize_lim,
                        'final_dim': final_dim,
                        # 'rot_lim': rot_lim,
                        # 'H': H, 'W': W,
                        # 'rand_flip': rand_flip,
                        # 'bot_pct_lim': bot_pct_lim,
                        # # 'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                 # # 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                        # 'cams': ['CAM_FRONT'],
                        # 'Ncams': ncams,
                    }
        
        xbound=[-32.0, 32.0, 0.5]
        ybound=[0.0, 104.0, 0.5]

        # xbound=[-5.0, 5.0, 0.5]
        # ybound=[0.0, 50.0, 0.5]
        # zbound=[-10.0, 10.0, 20.0]
        zbound=[-10.0, 10.0, 0.5]
        dbound=[4.0, 45.0, 1.0]

        grid_conf = {
            'xbound': xbound,
            'ybound': ybound,
            'zbound': zbound,
            'dbound': dbound,
        }
        # data_aug_conf -> final dim
        # outC needed for the bevEncode-> bevEncode removed
        # the grid conf is kind of important
        # num_y_steps defines the size of the the output
        model_geo = compile_model(grid_conf, data_aug_conf, outC=1, num_y_steps=args.num_y_steps, intrins=args.K)
        
        # define_init_weights(model_geo, args.weight_init)

        if not args.no_cuda:
            # Load model on gpu before passing params to optimizer
            model_seg = model_seg.cuda(0)
            model_geo = model_geo.cuda(0)
            

        # load segmentation model
        checkpoint = torch.load(pretrained_feat_model)
        model_seg = load_my_state_dict(model_seg, checkpoint['state_dict'])
        model_seg.eval()  # do not back propagate to model1

        # load geometry model
        best_test_name = glob.glob(os.path.join(args.save_path, 'model_best*'))[0]
        if os.path.isfile(best_test_name):
            sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
            print("=> loading checkpoint '{}'".format(best_test_name))
            checkpoint = torch.load(best_test_name)
            new_params = model_geo.state_dict().copy()
            saved_state_dict = checkpoint['state_dict']
            saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in new_params}
            # model_geo.load_state_dict(checkpoint['state_dict'])
            new_params.update(saved_state_dict) 
            model_geo.load_state_dict(new_params)
            
            
            

        
        else:
            print("=> no checkpoint found at '{}'".format(best_test_name))

        # Data loader
        
        # initialize visual saver
        vs_saver = Visualizer(args, args.vis_folder)

        mkdir_if_missing(os.path.join(args.save_path, 'example/' + args.vis_folder))
        
        print("=======>>result of input resolution W= {}, H= {}".format(args.resize_w , args.resize_h))
        
        test_dataset = LaneDataset(args.dataset_dir, test_gt_file, args)
        # assign std of valid dataset to be consistent with train dataset
        with open(ops.join(args.data_dir, 'geo_anchor_std.json')) as f:
            anchor_std = json.load(f)
        test_dataset.set_x_off_std(anchor_std['x_off_std'])
        if not args.no_3d:
            test_dataset.set_z_std(anchor_std['z_std'])
        test_dataset.normalize_lane_label()
        test_loader = get_loader(test_dataset, args)

        eval_stats = deploy(args, test_loader, test_dataset, model_seg, model_geo, vs_saver, test_gt_file, vis)


