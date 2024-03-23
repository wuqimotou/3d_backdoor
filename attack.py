import os
from pickle import FALSE
import sys
import numpy as np
# from collections import Iterable
import importlib
import open3d as o3d
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import os.path
import random
import torch
import numpy as np
import os
import argparse
import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils1.ModelNetDataLoader import ModelNetDataLoader
from data_utils1.ShapeNetDataLoader import PartNormalDataset
from torch.utils.data import DataLoader, TensorDataset


from utils.logging import Logging_str
from utils.utils import set_seed


from utils.set_distance import ChamferDistance, HausdorffDistance
from baselines import *


class PointCloudAttack(object):
    def __init__(self, args):
        """Shape-invariant Adversarial Attack for 3D Point Clouds.
        """
        self.args = args
        self.device = args.device

        self.eps = args.eps
        self.normal = args.normal
        self.step_size = args.step_size
        self.num_class = args.num_class
        self.max_steps = args.max_steps
        self.top5_attack = args.top5_attack

        assert args.transfer_attack_method is None or args.query_attack_method is None
        assert not args.transfer_attack_method is None or not args.query_attack_method is None
        self.attack_method = args.transfer_attack_method if args.query_attack_method is None else args.query_attack_method

        self.build_models()
        self.defense_method = args.defense_method
        if not args.defense_method is None:
            self.pre_head = self.get_defense_head(args.defense_method)


    def build_models(self):
        """Build white-box surrogate model and black-box target model.
        """
        # load white-box surrogate models
        MODEL = importlib.import_module(self.args.surrogate_model)
        wb_classifier = MODEL.get_model(
            self.num_class,
            normal_channel=self.normal
        )
        wb_classifier = wb_classifier.to(self.device)
        # load black-box target models
        MODEL = importlib.import_module(self.args.target_model)
        classifier = MODEL.get_model(
            self.num_class,
            normal_channel=self.normal
        )
        classifier = classifier.to(self.args.device)
        # load model weights
        wb_classifier = self.load_models(wb_classifier, self.args.surrogate_model)
        classifier = self.load_models(classifier, self.args.target_model)
        # set eval
        self.wb_classifier = wb_classifier.eval()
        self.classifier = classifier.eval()


    def load_models(self, classifier, model_name):
        """Load white-box surrogate model and black-box target model.
        """
        model_path = os.path.join('./checkpoint/' + self.args.dataset, model_name)
        if os.path.exists(model_path + '.pth'):
            checkpoint = torch.load(model_path + '.pth')
        elif os.path.exists(model_path + '.t7'):
            checkpoint = torch.load(model_path + '.t7')
        elif os.path.exists(model_path + '.tar'):
            checkpoint = torch.load(model_path + '.tar')
        else:
            raise NotImplementedError

        try:
            if 'model_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state_dict'])
            elif 'model_state' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state'])
            else:
                classifier.load_state_dict(checkpoint)
        except:
            classifier = nn.DataParallel(classifier)
            classifier.load_state_dict(checkpoint)
        return classifier

    def get_normal_vector(self, points):
        """Calculate the normal vector.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 3].
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.squeeze(0).detach().cpu().numpy())
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        normal_vec = torch.FloatTensor(pcd.normals).cuda().unsqueeze(0)
        return normal_vec


    def get_spin_axis_matrix(self, normal_vec):
        """Calculate the spin-axis matrix.

        Args:
            normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
        """
        _, N, _ = normal_vec.shape
        x = normal_vec[:,:,0] # [1, N]
        y = normal_vec[:,:,1] # [1, N]
        z = normal_vec[:,:,2] # [1, N]
        assert abs(normal_vec).max() <= 1
        u = torch.zeros(1, N, 3, 3).cuda()
        denominator = torch.sqrt(1-z**2) # \sqrt{1-z^2}, [1, N]
        u[:,:,0,0] = y / denominator
        u[:,:,0,1] = - x / denominator
        u[:,:,0,2] = 0.
        u[:,:,1,0] = x * z / denominator
        u[:,:,1,1] = y * z / denominator
        u[:,:,1,2] = - denominator
        u[:,:,2] = normal_vec
        # revision for |z| = 1, boundary case.
        pos = torch.where(abs(z ** 2 - 1) < 1e-4)[1]
        u[:,pos,0,0] = 1 / np.sqrt(2)
        u[:,pos,0,1] = - 1 / np.sqrt(2)
        u[:,pos,0,2] = 0.
        u[:,pos,1,0] = z[:,pos] / np.sqrt(2)
        u[:,pos,1,1] = z[:,pos] / np.sqrt(2)
        u[:,pos,1,2] = 0.
        u[:,pos,2,0] = 0.
        u[:,pos,2,1] = 0.
        u[:,pos,2,2] = z[:,pos]
        return u.data


    def get_transformed_point_cloud(self, points, normal_vec):
        """Calculate the spin-axis matrix.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 3].
            normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
        """
        intercept = torch.mul(points, normal_vec).sum(-1, keepdim=True) # P \cdot N, [1, N, 1]
        spin_axis_matrix = self.get_spin_axis_matrix(normal_vec) # U, [1, N, 3, 3]
        translation_matrix = torch.mul(intercept, normal_vec).data # (P \cdot N) N, [1, N, 3]
        new_points = points + translation_matrix #  P + (P \cdot N) N, [1, N, 3]
        new_points = new_points.unsqueeze(-1) # P + (P \cdot N) N, [1, N, 3, 1]
        new_points = torch.matmul(spin_axis_matrix, new_points) # P' = U (P + (P \cdot N) N), [1, N, 3, 1]
        new_points = new_points.squeeze(-1).data # P', [1, N, 3]
        return new_points, spin_axis_matrix, translation_matrix


    def get_original_point_cloud(self, new_points, spin_axis_matrix, translation_matrix):
        """Calculate the spin-axis matrix.

        Args:
            new_points (torch.cuda.FloatTensor): the transformed point cloud with N points, [1, N, 3].
            spin_axis_matrix (torch.cuda.FloatTensor): the rotate matrix for transformation, [1, N, 3, 3].
            translation_matrix (torch.cuda.FloatTensor): the offset matrix for transformation, [1, N, 3, 3].
        """
        inputs = torch.matmul(spin_axis_matrix.transpose(-1, -2), new_points.unsqueeze(-1)) # U^T P', [1, N, 3, 1]
        inputs = inputs - translation_matrix.unsqueeze(-1) # P = U^T P' - (P \cdot N) N, [1, N, 3, 1]
        inputs = inputs.squeeze(-1) # P, [1, N, 3]
        return inputs


    def run(self, points, target):
        """Main attack method.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        if self.attack_method == 'ifgm_ours':
            return self.shape_invariant_ifgm(points, target)

        elif self.attack_method == 'simba':
            return self.simba_attack(points, target)

        elif self.attack_method == 'simbapp':
            return self.simbapp_attack(points, target)

        elif self.attack_method == 'ours':
            return self.shape_invariant_query_attack(points, target)

        else:
            NotImplementedError




    def CWLoss(self, logits, target, kappa=0, tar=False, num_classes=40):
        """Carlini & Wagner attack loss.

        Args:
            logits (torch.cuda.FloatTensor): the predicted logits, [1, num_classes].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
        # print("logits.size(0)=")
        # print(logits.size(0))
        # print("target=" )
        # print(target)
        target_one_hot = Variable(torch.eye(num_classes).type(torch.cuda.FloatTensor)[target.long()].cuda())
        # print("targ_ont_hot=")
        # print(target_one_hot)
        real = torch.sum(target_one_hot*logits, 1)
        # print("reaj=")
        # print(real)
        # print("___________(1-target_one_hot)*logits = ")
        # print((1-target_one_hot)*logits)
        if not self.top5_attack:
            ### top-1 attack
            # other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
            other = ((1 - target_one_hot) * logits)[0, 26]
        else:
            ### top-5 attack
            other = torch.topk((1-target_one_hot)*logits - (target_one_hot*10000), 5)[0][:, 4]
        kappa = torch.zeros_like(other).fill_(kappa)
        # print(torch.max(other-real, kappa))

        if tar:
            return torch.sum(torch.max(other-real, kappa))
        else :
            return torch.sum(torch.max(real-other, kappa))


    def shape_invariant_ifgm(self, points, target):
        """Black-box I-FGSM based on shape-invariant sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        normal_vec = points[:,:,-3:].data # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].data # P, [1, N, 3]
        ori_points = points.data
        clip_func = ClipPointsLinf(budget=self.eps)# * np.sqrt(3*1024))

        for i in range(self.max_steps):
            # P -> P', detach()
            new_points, spin_axis_matrix, translation_matrix = self.get_transformed_point_cloud(points, normal_vec)
            new_points = new_points.detach()
            new_points.requires_grad = True
            # P' -> P
            points = self.get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
            points = points.transpose(1, 2) # P, [1, 3, N]
            # get white-box gradients
            if not self.defense_method is None:
                logits = self.wb_classifier(self.pre_head(points))
            else:
                logits = self.wb_classifier(points)
            loss = self.CWLoss(logits, target, kappa=0., tar=False, num_classes=self.num_class)
            self.wb_classifier.zero_grad()
            loss.backward()
            # print(loss.item(), logits.max(1)[1], target)
            grad = new_points.grad.data # g, [1, N, 3]
            grad[:,:,2] = 0.
            # update P', P and N
            # # Linf
            # new_points = new_points - self.step_size * torch.sign(grad)
            # L2
            norm = torch.sum(grad ** 2, dim=[1, 2]) ** 0.5
            new_points = new_points - self.step_size * np.sqrt(3*1024) * grad / (norm[:, None, None] + 1e-9)
            points = self.get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix) # P, [1, N, 3]
            points = clip_func(points, ori_points)
            # points = torch.min(torch.max(points, ori_points - self.eps), ori_points + self.eps) # P, [1, N, 3]
            normal_vec = self.get_normal_vector(points) # N, [1, N, 3]

        with torch.no_grad():
            adv_points = points.data
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.transpose(1, 2).detach()))
            else:
                adv_logits = self.classifier(points.transpose(1, 2).detach())
            adv_target = adv_logits.data.max(1)[1]
        # print(target)
        # print(adv_target)
        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1

        del normal_vec, grad, new_points, spin_axis_matrix, translation_matrix
        return adv_points, adv_target, (adv_logits.data.max(1)[1] != target).sum().item()


    def shape_invariant_query_attack(self, points, target):
        """Blaxk-box query-based attack based on point-cloud sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        normal_vec = points[:,:,-3:].data # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].data # P, [1, N, 3]
        ori_points = points.data
        # initialization
        query_costs = 0
        with torch.no_grad():
            points = points.transpose(1, 2)
            if not self.defense_method is None:
                adv_logits = self.classifier(self.pre_head(points.detach()))
            else:
                adv_logits = self.classifier(points)
            adv_target = adv_logits.max(1)[1]
            query_costs += 1
        # if categorized wrong
        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        if adv_target != target:
            return points.transpose(1, 2), adv_target, query_costs

        # P -> P', detach()
        points = points.transpose(1, 2)
        new_points, spin_axis_matrix, translation_matrix = self.get_transformed_point_cloud(points.detach(), normal_vec)
        new_points = new_points.detach()
        new_points.requires_grad = True

        # P' -> P
        inputs = self.get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix)
        inputs = torch.min(torch.max(inputs, ori_points - self.eps), ori_points + self.eps)
        inputs = inputs.transpose(1, 2) # P, [1, 3, N]

        # get white-box gradients
        logits = self.wb_classifier(inputs)
        loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
        self.wb_classifier.zero_grad()
        loss.backward()

        grad = new_points.grad.data # g, [1, N, 3]
        grad[:,:,2] = 0.
        new_points.requires_grad = False
        rankings = torch.sqrt(grad[:,:,0] ** 2 + grad[:,:,1] ** 2) # \sqrt{g_{x'}^2+g_{y'}^2}, [1, N]
        directions = grad / (rankings.unsqueeze(-1)+1e-16) # (g_{x'}/r,g_{y'}/r,0), [1, N, 3]

        # rank the sensitivity map in the desending order
        point_list = []
        for i in range(points.size(1)):
            point_list.append((i, directions[:,i,:], rankings[:,i].item()))
        sorted_point_list = sorted(point_list, key=lambda c: c[2], reverse=True)

        # query loop
        i = 0
        best_loss = -999.
        while best_loss < -0.5 and i < len(sorted_point_list):
            idx, direction, _ = sorted_point_list[i]
            for eps in {self.step_size, -self.step_size}:
                pert = torch.zeros_like(new_points).cuda()
                pert[:,idx,:] += eps * direction
                inputs = new_points + pert
                inputs = torch.matmul(spin_axis_matrix.transpose(-1, -2), inputs.unsqueeze(-1)) # U^T P', [1, N, 3, 1]
                inputs = inputs - translation_matrix.unsqueeze(-1) # P = U^T P' - (P \cdot N) N, [1, N, 3, 1]
                inputs = inputs.squeeze(-1).transpose(1, 2) # P, [1, 3, N]
                # inputs = torch.clamp(inputs, -1, 1)
                with torch.no_grad():
                    if not self.defense_method is None:
                        logits = self.classifier(self.pre_head(inputs.detach()))
                    else:
                        logits = self.classifier(inputs.detach()) # [1, num_class]
                    query_costs += 1
                loss = self.CWLoss(logits, target, kappa=-999., tar=True, num_classes=self.num_class)
                if loss.item() > best_loss:
                    # print(loss.item())
                    best_loss = loss.item()
                    new_points = new_points + pert
                    adv_target = logits.max(1)[1]
                    break
            i += 1
        # print(query_costs)
        # print(target)
        # print(adv_target)
        print(best_loss)
        adv_points = inputs.transpose(1, 2).data
        if self.top5_attack:
            target_top_5 = logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1
        del grad
        # return adv_points, adv_target, query_costs
        if best_loss < -0.8:
            return adv_points, -1, query_costs
        else:
            return adv_points, adv_target, query_costs


parser = argparse.ArgumentParser(description='Shape-invariant 3D Adversarial Point Clouds')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 1)')
parser.add_argument('--input_point_nums', type=int, default=1024,
                    help='Point nums of each point cloud')
parser.add_argument('--seed', type=int, default=2024, metavar='S',
                    help='random seed (default: 2024)')
parser.add_argument('--dataset', type=str, default='ModelNet40',
                    choices=['ModelNet40', 'ShapeNetPart'])
parser.add_argument('--data_path', type=str,
                    default='./data/modelnet40_normal_resampled/')
parser.add_argument('--normal', action='store_true', default=False,
                    help='Whether to use normal information [default: False]')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Worker nums of data loading.')

parser.add_argument('--transfer_attack_method', type=str, default=None,
                    choices=['ifgm_ours'])
parser.add_argument('--query_attack_method', type=str, default=None,
                    choices=['simbapp', 'simba', 'ours'])
parser.add_argument('--surrogate_model', type=str, default='pointnet_cls',
                    choices=['pointnet_cls', 'pointnet2_cls_msg', 'dgcnn', 'pointconv', 'pointcnn', 'paconv', 'pct', 'curvenet', 'simple_view'])
parser.add_argument('--target_model', type=str, default='dgcnn',
                    choices=['pointnet_cls', 'pointnet2_cls_msg', 'dgcnn', 'pointconv', 'pointcnn', 'paconv', 'pct', 'curvenet', 'simple_view'])
parser.add_argument('--defense_method', type=str, default=None,
                    choices=['sor', 'srs', 'dupnet'])
parser.add_argument('--top5_attack', action='store_true', default=False,
                    help='Whether to attack the top-5 prediction [default: False]')

parser.add_argument('--max_steps', default=50, type=int,
                    help='max iterations for black-box attack')
parser.add_argument('--eps', default=0.16, type=float,
                    help='epsilon of perturbation')
parser.add_argument('--step_size', default=0.32, type=float,
                    help='step-size of perturbation')
args = parser.parse_args()

# basic configuration
set_seed(args.seed)
args.device = torch.device("cuda")

def load_data(args):
    """Load the dataset from the given path.
    """
    print('Start Loading Dataset...')
    if args.dataset == 'ModelNet40':
        TEST_DATASET = ModelNetDataLoader(
            root=args.data_path,
            npoint=args.input_point_nums,
            split='test',
            normal_channel=True
        )
    elif args.dataset == 'ShapeNetPart':
        TEST_DATASET = PartNormalDataset(
            root=args.data_path,
            npoints=args.input_point_nums,
            split='test',
            normal_channel=True
        )
    else:
        raise NotImplementedError

    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print('Finish Loading Dataset...')
    return testDataLoader



def data_preprocess(data):
    """Preprocess the given data and label.
    """
    points, target = data

    points = points # [B, N, C]
    target = target[:, 0] # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target

def save_tensor_as_txt(points, filename):
    """Save the torch tensor into a txt file.
    """
    points = points.squeeze(0).detach().cpu().numpy()
    with open(filename, "a") as file_object:
        for i in range(points.shape[0]):
            msg = str(points[i][0]) + ',' + str(points[i][1]) + ',' + str(points[i][2])
            # msg = str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2]) + \
            #       ' ' + str(points[i][3].item()) + ' ' + str(points[i][3].item()) + ' ' + str(1 - points[i][3].item())
            file_object.write(msg + '\n')
        file_object.close()

    print('Have saved the tensor into {}'.format(filename))


if __name__ == '__main__':
    sys.path.append(r'.\models')
    test_loader = load_data(args)
    num_class = 0
    if args.dataset == 'ModelNet40':
        num_class = 40
    elif args.dataset == 'ShapeNetPart':
        num_class = 16
    assert num_class != 0
    args.num_class = num_class

    # load model
    attack = PointCloudAttack(args)

    # start attack
    atk_success = 0
    avg_query_costs = 0.
    avg_mse_dist = 0.
    avg_chamfer_dist = 0.
    avg_hausdorff_dist = 0.
    avg_time_cost = 0.
    tt = torch.zeros(40)
    chamfer_loss = ChamferDistance()
    hausdorff_loss = HausdorffDistance()
    save_id = 0
    #清空之前的后门样本
    shutil.rmtree('./backdoorSample')
    os.mkdir('./backdoorSample')
    f = open('./backdoor.txt', 'a')
    f.truncate(0)
    f.close()
    for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        # prepare data for testing
        points, target = data_preprocess(data)
        target = target.long()

        # start attack
        t0 = time.process_time()
        adv_points, adv_target, query_costs = attack.run(points, target)
        if adv_target == -1:
            print("不合格的后门样本！")
            continue
        #保存用于后门攻击的后门样本点云
        save_tensor_as_txt(adv_points, "./backdoorSample/plant_0%d.txt"%(save_id + 341))
        #将生成的后门样本名称组成列表
        f = open('./backdoor.txt', 'a')
        f.write("plant_0" + str(save_id + 341) + '\n')
        f.close()
        save_id += 1

        # if adv_target == 26:
        #     print(batch_id)
        #     print("成功！")
        # else:
        #     print(batch_id)
        #     print("失败！")

        print("原来的类别为：", target)
        print("对抗的类别为：", adv_target)
        if target == 0 :
            tt[adv_target.cpu()] += 1
            print(tt)
        t1 = time.process_time()
        avg_time_cost += t1 - t0
        if not args.query_attack_method is None:
            print('>>>>>>>>>>>>>>>>>>>>>>>')
            print('Query cost: ', query_costs)
            print('>>>>>>>>>>>>>>>>>>>>>>>')
            avg_query_costs += query_costs
        # atk_success += 1 if adv_target != target else 0
        atk_success += 1 if adv_target == 26 else 0
        print(atk_success)

        # modified point num count
        points = points[:,:,:3].data # P, [1, N, 3]
        pert_pos = torch.where(abs(adv_points-points).sum(2))
        count_map = torch.zeros_like(points.sum(2))
        count_map[pert_pos] = 1.
        # print('Perturbed point num:', torch.sum(count_map).item())

        avg_mse_dist += np.sqrt(F.mse_loss(adv_points, points).detach().cpu().numpy() * 3072)
        avg_chamfer_dist += chamfer_loss(adv_points, points)
        avg_hausdorff_dist += hausdorff_loss(adv_points, points)

    atk_success /= batch_id + 1
    print('Attack success rate: ', atk_success)
    avg_time_cost /= batch_id + 1
    print('Average time cost: ', avg_time_cost)
    if not args.query_attack_method is None:
        avg_query_costs /= batch_id + 1
        print('Average query cost: ', avg_query_costs)
    avg_mse_dist /= batch_id + 1
    print('Average MSE Dist:', avg_mse_dist)
    avg_chamfer_dist /= batch_id + 1
    print('Average Chamfer Dist:', avg_chamfer_dist.item())
    avg_hausdorff_dist /= batch_id + 1
    print('Average Hausdorff Dist:', avg_hausdorff_dist.item())