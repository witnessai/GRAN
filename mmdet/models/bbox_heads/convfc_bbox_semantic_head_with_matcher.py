# coding: utf-8
import torch.nn as nn
import torch

from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head_semantic import BBoxSemanticHead

import ipdb
import numpy as np
import torch.nn.functional as F
from ..losses import accuracy
from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from torch.nn import init
import math
class XavierLinear(nn.Module):
    '''
    Simple Linear layer with Xavier init

    Paper by Xavier Glorot and Yoshua Bengio (2010):
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, dim_in_hid_out, act_fn='ReLU', last_act=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(dim_in_hid_out) - 1):
            layers.append(XavierLinear(dim_in_hid_out[i], dim_in_hid_out[i + 1]))
            if i < len(dim_in_hid_out) - 2 or last_act:
                layers.append(getattr(torch.nn, act_fn)())
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

@HEADS.register_module
class ConvFCSemanticBBoxHead_with_matcher(BBoxSemanticHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_semantic_convs=0,
                 num_semantic_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 semantic_dims=300,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCSemanticBBoxHead_with_matcher, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_semantic_convs +
                num_semantic_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_semantic_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_semantic:
            assert num_semantic_convs == 0 and num_semantic_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        
        self.num_shared_convs = num_shared_convs # 0
        self.num_shared_fcs = num_shared_fcs # 2
        self.num_semantic_convs = num_semantic_convs # 0
        self.num_semantic_fcs = num_semantic_fcs # 0
        self.num_reg_convs = num_reg_convs # 0
        self.num_reg_fcs = num_reg_fcs # 0
        self.conv_out_channels = conv_out_channels # 256
        self.fc_out_channels = 300 #fc_out_channels//2# 1024
        self.conv_cfg = conv_cfg # None
        self.norm_cfg = norm_cfg # None

        # add shared convs and fcs
        # morjio output:
        # (ModuleList(), ModuleList(
        # (0): Linear(in_features=12544, out_features=1024, bias=True)
        # (1): Linear(in_features=1024, out_features=1024, bias=True)
        # ), 1024)
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim # 1024

        # add semantic specific branch
        # morjio output:
        # (ModuleList(), ModuleList(), 1024)
        self.semantic_convs, self.semantic_fcs, self.semantic_last_dim = \
            self._add_conv_fc_branch(
                self.num_semantic_convs, self.num_semantic_fcs, self.shared_out_channels)

        # add reg specific branch
        # morjio output:
        # (ModuleList(), ModuleList(), 1024)
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_semantic_fcs == 0:
                self.semantic_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_semantic and fc_reg since input channels are changed
        if self.with_semantic: # True
            self.fc_semantic = nn.Linear(self.semantic_last_dim, semantic_dims) # 1024, 300
            if self.voc is not None:
                self.kernel_semantic = nn.Linear(self.voc.shape[1], self.vec.shape[0])  # n*300
            else:
                self.kernel_semantic = nn.Linear(self.vec.shape[1], self.vec.shape[1])

        if self.with_reg and self.reg_with_semantic: # self.reg_with_semantic is False
            self.fc_reg_sem = nn.Linear(self.reg_last_dim, semantic_dims)
            if not self.share_semantic:
                self.kernel_semantic_reg = nn.Linear(self.voc.shape[1], self.vec.shape[0])
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.num_classes, out_dim_reg)

        if self.with_reg and not self.reg_with_semantic:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg) # 1024, 4

        self.fc_res = nn.Linear(self.vec.shape[0], self.vec.shape[0])
        # self.fc_res = nn.Linear(self.semantic_last_dim, self.vec.shape[0])

        self.has_matcher = True 
        # self.ext_kb = _ext_kb[:66, :66]
        self.ext_kb = np.ones((self.num_classes, self.num_classes))
        
        hidden_dim =  self.fc_out_channels
        self.fc_init_ont_ent = nn.Linear(300, hidden_dim)
        self.fc_mp_send_ont_ent = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_img_ent = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_ont_ent = MLP([2*hidden_dim//4, 2*hidden_dim//4, hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_img_ent = MLP([2*hidden_dim//4, 2*hidden_dim//4, hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_eq3_w_ont_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_ont_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_ont_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_ont_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_ont_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_ont_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_w_img_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_img_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_img_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_img_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_img_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_img_ent = nn.Linear(hidden_dim, hidden_dim)
        # self.fc_output_proj_img_ent = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
        # self.fc_output_proj_ont_ent = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False) 
        self.fc_output_proj_img_ent = MLP([hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
        self.fc_output_proj_ont_ent = MLP([hidden_dim, hidden_dim], act_fn='ReLU', last_act=False) 
        
        # for test using updated vec(word embedding)
        self.vec_update = self.vec

        self.count = 0

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCSemanticBBoxHead_with_matcher, self).init_weights()
        for module_list in [self.shared_fcs, self.semantic_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, rois, res_feats=None, context_feats=None, return_feats=False, resturn_center_feats=False):
        # shared part
        # morjio add:
        # x.shape is [1024, 256, 7, 7]
        if self.num_shared_convs > 0: # False, self.num_shared_convs is 0
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0: # self.num_shared_fcs is 2
            if self.with_avg_pool: # False
                x = self.avg_pool(x)
            # after op x.shape is [1024, 12544]
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        #x.shape is [1024, 1024]
        # separate branches
        x_semantic = x
        x_reg = x

        for conv in self.semantic_convs: # None
            x_semantic = conv(x_semantic)
        if x_semantic.dim() > 2: # False
            if self.with_avg_pool:
                x_semantic = self.avg_pool(x_semantic)
            x_semantic = x_semantic.view(x_semantic.size(0), -1)
        for fc in self.semantic_fcs: # None
            x_semantic = self.relu(fc(x_semantic))

        for conv in self.reg_convs: # None
            x_reg = conv(x_reg)
        if x_reg.dim() > 2: # False
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs: # None
            x_reg = self.relu(fc(x_reg))

        if self.with_semantic: # True
            # semantic_feature is [1024, 300]
            semantic_feature = self.fc_semantic(x_semantic)
            # ipdb.set_trace()
            if self.voc is not None: # True
                semantic_score = self.kernel_semantic(self.voc)
                if res_feats is not None: # False
                    semantic_score += self.fc_res(res_feats)
                res_feats = semantic_score

                semantic_score = torch.mm(semantic_score, self.vec)
                semantic_score = torch.tanh(semantic_score)
                semantic_score = torch.mm(semantic_feature, semantic_score) 
                # semantic_score is [1024, 66]
                # 1024*49
                if context_feats is not None:
                    semantic_score = torch.mul(semantic_score, context_feats)
            else:
                semantic_score = self.kernel_semantic(self.vec)
                semantic_score = torch.tanh(semantic_score)
                semantic_score = torch.mm(semantic_feature, semantic_score)
        else:
            semantic_score = None
        if self.with_reg and not self.reg_with_semantic: # True
            # bbox_pred shape [1024, 4]
            bbox_pred = self.fc_reg(x_reg)
        elif self.with_reg and self.reg_with_semantic: 
            semantic_reg_feature = self.fc_reg_sem(x_reg)
            if not self.share_semantic:
                semantic_reg_score = torch.mm(self.kernel_semantic_reg(self.voc), self.vec)
            else:
                semantic_reg_score = torch.mm(self.kernel_semantic(self.voc), self.vec)
            semantic_reg_score = torch.tanh(semantic_reg_score)
            semantic_reg_score = torch.mm(semantic_reg_feature, semantic_reg_score)
            bbox_pred = self.fc_reg(semantic_reg_score)
        else:
            bbox_pred = None

        # x.shape is [1024, 1024], self.ext_kb.shape is [66, 66]
        # self.vec.shape is [300, 66]
        match_sem_score = []
        self.count += 1
        
        if self.training:
            img_num = int(rois[-1, 0])+1
            clip_step = []
            for i in range(img_num):
                num_roi = torch.sum(rois[:, 0]==i)
                clip_step.append(int(num_roi))
            # ipdb.set_trace()
            # clip_step = 512
            # if  x.shape[0] % 512 != 0:   
            #     ipdb.set_trace()
            # img_num = x.shape[0]//clip_step  
        else:
            # clip_step = 1000
            # img_num = math.ceil(x.shape[0]/clip_step)
            clip_step = []
            img_num = int(rois[-1, 0])+1
            clip_step.append(x.shape[0])
        
        time_step_num = 3
        for i in range(img_num):
            edges_ont_ent2ent = torch.from_numpy(self.ext_kb).float().cuda() # shape is [66, 66]
            nodes_ont_ent = self.fc_init_ont_ent(self.vec.t()) # nodes_ont_ent.shape is [66, hidden_dim] = [66, 300] x [300, hidden_dim] 
            # nodes_ont_ent = self.vec.t() # remove self.fc_init_ont_ent
            edges_img_ent2ent = torch.ones(clip_step[i], clip_step[i]).cuda()

            # nodes_img_ent = semantic_feature[i*clip_step:(i+1)*clip_step, :]
            nodes_img_ent = x[i*clip_step[i]:(i+1)*clip_step[i], :]
            # ipdb.set_trace()
            edges_img2ont_ent = semantic_score[i*clip_step[i]:(i+1)*clip_step[i], :]
            edges_ont2img_ent = edges_img2ont_ent.t()
            for t in range(time_step_num):
                message_send_ont_ent = self.fc_mp_send_ont_ent(nodes_ont_ent) # [66, hidden_dim/4 ] = [66, hidden_dim] x MLP         (hidden_dim, hidden_dim/2, hidden_dim/4)
                message_send_img_ent = self.fc_mp_send_img_ent(nodes_img_ent) # [512, hidden_dim/4] = [512, hidden_dim] x MLP(hidden_dim, hidden_dim/2, hidden_dim/4)
                # message_received_ont_ent.shape is [66, hidden_dim]
                # ipdb.set_trace()
                message_received_ont_ent = self.fc_mp_receive_ont_ent(
                    torch.cat(  # torch.cat([66, hidden_dim/4], [66, hidden_dim/4])
                        [torch.mm(edges_ont_ent2ent.t(), message_send_ont_ent)] +
                        [torch.mm(edges_img2ont_ent.t(), message_send_img_ent)]
                    , 1)
                )
                # message_received_img_ent.shape is [512, hidden_dim]
                message_received_img_ent = self.fc_mp_receive_img_ent(
                    torch.cat( # torch.cat([clip_step, hidden_dim/4], [clip_step,  hidden_dim/4])
                        [torch.mm(edges_img_ent2ent.t(), message_send_img_ent)] +
                        [torch.mm(edges_ont2img_ent.t(), message_send_ont_ent)]
                    , 1)
                )
        
                z_ont_ent = torch.sigmoid(self.fc_eq3_w_ont_ent(message_received_ont_ent) + self.fc_eq3_u_ont_ent(nodes_ont_ent))
                r_ont_ent = torch.sigmoid(self.fc_eq4_w_ont_ent(message_received_ont_ent) + self.fc_eq4_u_ont_ent(nodes_ont_ent))
                h_ont_ent = torch.tanh(self.fc_eq5_w_ont_ent(message_received_ont_ent) + self.fc_eq5_u_ont_ent(r_ont_ent * nodes_ont_ent))
                nodes_ont_ent_new = (1 - z_ont_ent) * nodes_ont_ent + z_ont_ent * h_ont_ent


                z_img_ent = torch.sigmoid(self.fc_eq3_w_img_ent(message_received_img_ent) + self.fc_eq3_u_img_ent(nodes_img_ent))
                r_img_ent = torch.sigmoid(self.fc_eq4_w_img_ent(message_received_img_ent) + self.fc_eq4_u_img_ent(nodes_img_ent))
                h_img_ent = torch.tanh(self.fc_eq5_w_img_ent(message_received_img_ent) + self.fc_eq5_u_img_ent(r_img_ent * nodes_img_ent))
                nodes_img_ent_new = (1 - z_img_ent) * nodes_img_ent + z_img_ent * h_img_ent

                nodes_ont_ent = nodes_ont_ent_new 
                nodes_img_ent = nodes_img_ent_new 

                ent_cls_logits = torch.mm(self.fc_output_proj_img_ent(nodes_img_ent), self.fc_output_proj_ont_ent(nodes_ont_ent).t())
                # ent_cls_logits = torch.mm(nodes_img_ent, nodes_ont_ent.t())

                edges_img2ont_ent = F.softmax(ent_cls_logits, dim=1)
                edges_ont2img_ent = edges_img2ont_ent.t()
                self.vec_update = nodes_ont_ent.t()
            match_sem_score.append(ent_cls_logits) 
            
            # match_sem_score.append(edges_img2ont_ent)
        # ipdb.set_trace()
        match_sem_score = torch.cat(match_sem_score, 0)
        
        if return_feats:
            return semantic_score, bbox_pred, res_feats
        # elif resturn_center_feats:
        #     return semantic_score, bbox_pred, center_feats
        else:
            if self.has_matcher and self.training:
                return semantic_score, bbox_pred, match_sem_score
            else:
                # ipdb.set_trace()
                return match_sem_score, bbox_pred
    @force_fp32(apply_to=('semantic_score', 'bbox_pred'))
    def loss(self,
             semantic_score,
             bbox_pred,
             match_sem_score, 
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if semantic_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            # losses['loss_semantic'] = self.loss_semantic(
            #     semantic_score,
            #     labels,
            #     label_weights,
            #     avg_factor=avg_factor,
            #     reduction_override=reduction_override)
            # losses['semantic_acc'] = accuracy(semantic_score, labels)
        if match_sem_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_match'] = self.loss_semantic(
                match_sem_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['match_acc'] = accuracy(match_sem_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        

        return losses
    
    @force_fp32(apply_to=('semantic_score', 'bbox_pred'))
    def get_det_bboxes(self,
                       rois,
                       semantic_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(semantic_score, list):
            semantic_score = sum(semantic_score) / float(len(semantic_score))
        scores = F.softmax(semantic_score, dim=1) if semantic_score is not None else None
        # scores = LSoftmaxLinear(semantic_score, dim=1) if semantic_score is not None else None
        # ipdb.set_trace()
        update_flag = False
        # ipdb.set_trace()
        if update_flag:
            semantic_vector = self.vec_update
        else:
            semantic_vector = self.vec
        if self.gzsd:
        # if True
            
            seen_scores = torch.mm(scores, semantic_vector.t())
            seen_scores = torch.mm(seen_scores, semantic_vector)
            seen_bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                     self.target_stds, img_shape)

            unseen_scores = torch.mm(scores, semantic_vector.t())
            unseen_scores = torch.mm(unseen_scores, self.vec_unseen)
            unseen_bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                     self.target_stds, img_shape)

            if rescale: # True
                if isinstance(scale_factor, float):
                    seen_bboxes /= scale_factor
                    unseen_bboxes /= scale_factor
                else:
                    seen_bboxes /= torch.from_numpy(scale_factor).to(seen_bboxes.device)
                    unseen_bboxes /= torch.from_numpy(scale_factor).to(unseen_bboxes.device)

            if cfg is None: # False
                return [seen_bboxes, unseen_bboxes], [seen_scores, unseen_scores]
            else:
                seen_det_bboxes, seen_det_labels = multiclass_nms(seen_bboxes, seen_scores,
                                                        0.2, cfg.nms,
                                                        cfg.max_per_img)
                unseen_det_bboxes, unseen_det_labels = multiclass_nms(unseen_bboxes, unseen_scores,
                                                                  0.05, cfg.nms,
                                                                  cfg.max_per_img)
                # unseen_det_labels += 65
                # unseen_det_labels += 48
                unseen_det_labels += (self.num_classes - 1)

                det_bboxes = torch.cat([seen_det_bboxes, unseen_det_bboxes], dim=0)
                det_labels = torch.cat([seen_det_labels, unseen_det_labels], dim=0)
                # return [seen_det_bboxes, unseen_det_bboxes], [seen_det_labels, unseen_det_labels]
                return det_bboxes, det_labels

        if self.seen_class:
            scores = torch.mm(scores, semantic_vector.t())
            scores = torch.mm(scores, semantic_vector)
        # TODO ZSD  open these lines when unseen inference
        if not self.seen_class:
            scores = torch.mm(scores, semantic_vector.t())
            scores = torch.mm(scores, self.vec_unseen)

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:  
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels


@HEADS.register_module
class SharedFCSemanticBBoxHead_with_matcher(ConvFCSemanticBBoxHead_with_matcher):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCSemanticBBoxHead_with_matcher, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_semantic_convs=0,
            num_semantic_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


