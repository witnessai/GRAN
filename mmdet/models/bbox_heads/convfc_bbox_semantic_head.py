import torch.nn as nn
import torch

from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head_semantic import BBoxSemanticHead

import ipdb
@HEADS.register_module
class ConvFCSemanticBBoxHead(BBoxSemanticHead):
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
        super(ConvFCSemanticBBoxHead, self).__init__(*args, **kwargs)
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
        self.fc_out_channels = 512 #fc_out_channels # 1024
        self.conv_cfg = conv_cfg # None
        self.norm_cfg = norm_cfg # None
        # ipdb.set_trace()
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
            # self.vec.shape is [300, 66], self.voc.shape is [300, 4717]
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
        self.has_matcher = False

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
        super(ConvFCSemanticBBoxHead, self).init_weights()
        for module_list in [self.shared_fcs, self.semantic_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, res_feats=None, context_feats=None, return_feats=False, resturn_center_feats=False):
        # shared part
        # morjio add:
        # x.shape is [1024, 256, 7, 7]
        if self.num_shared_convs > 0: # self.num_shared_convs is 0
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
        if return_feats:
            return semantic_score, bbox_pred, res_feats
        # elif resturn_center_feats:
        #     return semantic_score, bbox_pred, center_feats
        else:
            return semantic_score, bbox_pred


@HEADS.register_module
class SharedFCSemanticBBoxHead(ConvFCSemanticBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCSemanticBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_semantic_convs=0,
            num_semantic_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
