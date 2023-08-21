# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_
from adet.utils.misc import inverse_sigmoid
from adet.modeling.srformer.utils import MLP, gen_point_pos_embed
from .ms_deform_attn import MSDeformAttn
from timm.models.layers import DropPath
from .feature_fuse import ScaleFeatureSelection


class DeformableTransformer_Det(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            return_intermediate_dec=False,
            num_feature_levels=4,
            dec_n_points=4,
            enc_n_points=4,
            num_proposals=100,
            num_ctrl_points=16,
            num_seg_layers=2,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_proposals = num_proposals
        self.total_num_feature_levels = 4
        

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        self.feature_fuse = ScaleFeatureSelection()

        decoder_layer = DeformableTransformerDecoderLayer_Det(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points
        )
        self.decoder_new = DeformableTransformerDecoder_AnchorDet(
            decoder_layer,
            num_decoder_layers,
            return_intermediate_dec,
            d_model,
            num_seg_layers
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.bbox_class_embed = None
        self.bbox_embed = None
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)

        self.semantic_seg_head = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model // 4, 3, padding=1),
            nn.GroupNorm(32, self.d_model // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model // 4, self.d_model // 4, 3, padding=1),
            nn.GroupNorm(32, self.d_model // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model // 4, 1, 3, padding=1))

        self.num_ctrl_points = num_ctrl_points

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 64
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 256
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_control_points_from_anchor(self, reference_points_anchor):
        # reference_points_anchor: bs, nq, 4
        # return size:
        # - reference_points: (bs, nq, n_pts, 2)
        assert reference_points_anchor.shape[-1] == 4
        reference_points = reference_points_anchor[:, :, None, :].repeat(1, 1, self.num_ctrl_points, 1)
        pts_per_side = self.num_ctrl_points // 2
        reference_points[:, :, 0, 0].sub_(reference_points[:, :, 0, 2] / 2)
        reference_points[:, :, 1:pts_per_side, 0] = reference_points[:, :, 1:pts_per_side, 2] / (pts_per_side - 1)
        reference_points[:, :, :pts_per_side, 0] = torch.cumsum(reference_points[:, :, :pts_per_side, 0], dim=-1)
        reference_points[:, :, pts_per_side:, 0] = reference_points[:, :, :pts_per_side, 0].flip(dims=[-1])
        reference_points[:, :, :pts_per_side, 1].sub_(reference_points[:, :, :pts_per_side, 3] / 2)
        reference_points[:, :, pts_per_side:, 1].add_(reference_points[:, :, pts_per_side:, 3] / 2)
        reference_points = torch.clamp(reference_points[:, :, :, :2], 0, 1)

        return reference_points

    def forward(self, srcs, masks, pos_embeds, query_embed, bk_feature=None):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten
        )

        # Rebuild Feature Maps form Encoder Queries
        split_size_or_sections = [None] * self.total_num_feature_levels
        for i in range(self.total_num_feature_levels):
            if i < self.total_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = memory.shape[1] - level_start_index[i]
        y = torch.split(memory, split_size_or_sections, dim=1)


        encoder_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            encoder_features.append(z.transpose(1, 2).reshape(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))


        multiscale_feature = self.feature_fuse(encoder_features, bk_feature)
        semantic_seg_mask = self.semantic_seg_head(multiscale_feature[0])


        src_flatten = []
        for idx in range(self.total_num_feature_levels):
            bs, c , h, w = multiscale_feature[idx].shape
            src_flatten.append(multiscale_feature[idx].flatten(2).transpose(1, 2))
        memory = torch.cat(src_flatten, 1)
        
        # prepare input for decoder
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

        
        enc_outputs_class = self.bbox_class_embed(output_memory)
        enc_outputs_anchor_unact = self.bbox_embed(output_memory) + output_proposals
        topk = self.num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]

        
        tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
        topk_anchors_unact = torch.gather(enc_outputs_anchor_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_anchors_unact = topk_anchors_unact.detach()
        reference_points = topk_anchors_unact.sigmoid()  # (bs, nq, 4)
        reference_points = self.init_control_points_from_anchor(reference_points)
        
        init_reference_out = reference_points
        query_embed = tgt_undetach.detach()[:,:,None] + query_embed.unsqueeze(0).expand(bs, -1, -1, -1) # (bs, nq,1, 256)


        inter_references, segmentation_map, inter_class, lower_level_masks = self.decoder_new(
            query_embed,# (bs, nq,1, 256)
            reference_points, # (bs, nq,1, 2)
            memory, #(bs, map_size, 256)
            spatial_shapes,
            level_start_index,
            valid_ratios,
            src_padding_mask = mask_flatten,
            query_pos = None,
            fused_feature = multiscale_feature,
            semantic_mask = semantic_seg_mask,
            level_embed = self.level_embed
        )
        inter_references_out = inter_references
        

        return inter_references_out, enc_outputs_class, enc_outputs_anchor_unact, segmentation_map, semantic_seg_mask, inter_class, lower_level_masks


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model=256,
            d_ffn=1024,
            dropout=0.1,
            activation="relu",
            n_levels=4,
            n_heads=8,
            n_points=4
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class CirConv(nn.Module):
    def __init__(self, d_model, n_adj=4):
        super(CirConv, self).__init__()
        self.n_adj = n_adj
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=self.n_adj*2+1)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(d_model)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, tgt):
        shape = tgt.shape
        tgt = (tgt.flatten(0, 1)).permute(0,2,1).contiguous()  # (bs*nq, dim, n_pts)
        tgt = torch.cat([tgt[..., -self.n_adj:], tgt, tgt[..., :self.n_adj]], dim=2)
        tgt = self.relu(self.norm(self.conv(tgt)))
        tgt = tgt.permute(0,2,1).contiguous().reshape(shape)
        return tgt


class DeformableTransformerDecoderLayer_Det(nn.Module):
    def __init__(
            self,
            d_model=256,
            d_ffn=1024,
            dropout=0.1,
            activation="relu",
            n_levels=4,
            n_heads=8,
            n_points=4,
    ):
        super().__init__()


        # cross attention
        self.attn_cross = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout_cross = nn.Dropout(dropout)
        self.norm_cross = nn.LayerNorm(d_model)

        # intra-group self-attention
        self.attn_intra = nn.MultiheadAttention(d_model, n_heads, dropout=0.)
        self.circonv = CirConv(d_model)
        self.norm_fuse = nn.LayerNorm(d_model)
        self.mlp_fuse = nn.Linear(d_model, d_model)
        self.drop_path = DropPath(0.1)
        self.norm_intra = nn.LayerNorm(d_model)

        # inter-group self-attention
        self.attn_inter = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_inter = nn.Dropout(dropout)
        self.norm_inter = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
            self,
            tgt,
            query_pos,
            query_pos_anchor,
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask=None,
            lid=0
    ):
        # input size
        # - tgt:        (bs, n_q, 1, dim)
        # - query_pos:  #  (bs, nq, 1, dim)

        # intra-group self-attention
        shortcut = tgt
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt = self.attn_intra(
            q.flatten(0, 1).transpose(0, 1),
            k.flatten(0, 1).transpose(0, 1),
            tgt.flatten(0, 1).transpose(0, 1),
        )[0].transpose(0, 1).reshape(q.shape)
       
        tgt_circonv = self.drop_path(self.circonv(shortcut+query_pos))
        
        tgt = shortcut + self.norm_intra(self.drop_path(tgt) + tgt_circonv)
        tgt = tgt + self.drop_path(self.norm_fuse(self.mlp_fuse(tgt)))

        # inter-group self-attention
        q_inter = k_inter = torch.swapdims(self.with_pos_embed(tgt, query_pos_anchor), 1, 2)
        tgt_inter = torch.swapdims(tgt, 1, 2)  # (bs, 1, n_q, dim)
        tgt2_inter = self.attn_inter(
            q_inter.flatten(0, 1).transpose(0, 1),
            k_inter.flatten(0, 1).transpose(0, 1),
            tgt_inter.flatten(0, 1).transpose(0, 1)
        )[0].transpose(0, 1).reshape(q_inter.shape)
        tgt_inter = tgt_inter + self.dropout_inter(tgt2_inter)
        tgt_inter = torch.swapdims(self.norm_inter(tgt_inter), 1, 2) # (bs,  n_q,1, dim)

        reference_points_loc = reference_points  #(bs, nq, 1, 4, 2)
        tgt2 = self.attn_cross(
            self.with_pos_embed(tgt_inter, query_pos).flatten(1, 2),
            reference_points_loc.flatten(1, 2),
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask
        ).reshape(tgt_inter.shape)
        tgt_inter = tgt_inter + self.dropout_cross(tgt2)
        tgt = self.norm_cross(tgt_inter)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder_AnchorDet(nn.Module):
    def __init__(
            self,
            decoder_layer,
            num_layers,
            return_intermediate=False,
            d_model=256,
            num_seg_layers=2
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.ctrl_point_coord = None
        self.decoder_norm = None
        self.mask_embed = None
        self.device = None
        self.mask_feature_proj = None
        self.offset_embed = None
        self.anchor_offset_embed = None
        self.query_aggregation_weights = None
        self.ctrl_point_class = None
        self.ctrl_point_embed = None
        self.num_seg_layers = num_seg_layers

        self.ref_point_head = MLP(d_model, d_model, d_model, 2)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=1024, batch_first=True)
        self.encoder_layer = nn.ModuleList([self.encoder_layer for _ in range(self.num_seg_layers)])

        self.feature_proj = nn.Linear(d_model, d_model)
        self.feature_proj = nn.ModuleList([self.feature_proj for _ in range(self.num_seg_layers)])
        


    def forward_prediction_heads_class(self, output, query_pos, lvl):
        outputs_class = self.ctrl_point_class[lvl](output).mean(-2)

        return outputs_class
        

    def forward_prediction_reg_only(self, output, reference_point, lvl):
        corrd_offset = self.offset_embed[lvl](output)
        reference_point_updated = (inverse_sigmoid(reference_point) + corrd_offset).sigmoid()

        return reference_point_updated
    

    def forward_prediction_reg_and_seg(self, output, mask_features_ori, query_pos, semantic_mask, last_reference_points, level_embed, lvl):
        
        
        # Text Instance Mask Prediction
        b, nq, n_ctrl, h = output.shape
        query_aggregation_weights = self.query_aggregation_weights(output + query_pos).reshape(b,nq,n_ctrl,1)
        aggregated_query = self.decoder_norm((output * query_aggregation_weights).sum(dim=-2))
        mask_embed = self.mask_embed[lvl](aggregated_query)
        mask_features = self.mask_feature_proj(mask_features_ori[0])
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)


        # Mask-guided Query Enhancement
        instance_text_features = []
        lower_level_masks = []

        semantic_mask = semantic_mask.sigmoid().detach()
        ROI = F.softmax(outputs_mask.unsqueeze(2).flatten(-2), dim=1).detach() * semantic_mask.unsqueeze(2).flatten(-2)

        instance_text_feature = torch.sum(ROI * mask_features_ori[0].unsqueeze(1).flatten(-2), dim=-1) / ROI.sum(-1)
        instance_text_features.append(instance_text_feature)

        for i in range(1,4):
            lower_level_mask_features = self.mask_feature_proj(mask_features_ori[i])
            lower_level_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, lower_level_mask_features)
            lower_level_masks.append(lower_level_mask)

            ROI = F.softmax(lower_level_mask.unsqueeze(2).flatten(-2), dim=1).detach() *\
                  F.interpolate(semantic_mask, size=lower_level_mask.shape[-2:], mode="nearest").unsqueeze(2).flatten(-2)
            
            instance_text_feature = torch.sum(ROI * mask_features_ori[i].unsqueeze(1).flatten(-2), dim=-1) / ROI.sum(-1)
            instance_text_features.append(instance_text_feature)
        
        instance_text_features = torch.stack(instance_text_features).mean(dim=0)
        ROI = semantic_mask.unsqueeze(2).flatten(-2)
        global_text_feature = torch.sum(ROI * mask_features_ori[0].unsqueeze(1).flatten(-2), dim=-1) / ROI.sum(-1)

        instance_text_features = self.encoder_layer[lvl](instance_text_features) + global_text_feature
        instance_text_features = self.feature_proj[lvl](instance_text_features[:,:,None])


        # Mask as anchor prior
        b, nq, h, w = outputs_mask.shape
        x, y = torch.meshgrid(torch.linspace(1 / (w + 1),w / (w + 1),w),\
                              torch.linspace(1 / (h + 1),h / (h + 1),h))
        pre_defined_anchor_grids = torch.stack((x, y), 2).transpose(0,1).float().to(self.device)

        outputs_anchor_weights_map = F.softmax(outputs_mask.reshape(b,nq,h*w) / 0.3, dim=-1).reshape(b,nq,h,w).unsqueeze(-1)
        anchor_point = (pre_defined_anchor_grids.unsqueeze(0) * outputs_anchor_weights_map).sum(dim=(2,3)).detach() 
        

        # Polygon contrl points regression
        anchor_offset = self.anchor_offset_embed[lvl](output)
        reference_point_anchor = (inverse_sigmoid(anchor_point.unsqueeze(2)) + anchor_offset).sigmoid()
        
        corrd_offset = self.offset_embed[lvl](output)
        reference_point_updated = (inverse_sigmoid(last_reference_points) + corrd_offset).sigmoid()
        
        reference_point_updated = (reference_point_updated + reference_point_anchor) / 2
        

        return reference_point_updated, outputs_mask, lower_level_masks, instance_text_features#, anchor_point

    def forward(
            self,
            tgt,
            reference_points,
            src,
            src_spatial_shapes,
            src_level_start_index,
            src_valid_ratios,
            query_pos=None,
            src_padding_mask=None,
            fused_feature=None,
            semantic_mask = None,
            level_embed = None
    ):
        output = tgt  # bs, K, 16, 256
        intermediate_reference_points = []
        intermediate_segmentation_map = []
        intermediate_class = []

        for lid, layer in enumerate(self.layers):
            
            # Update queries with MQE
            if lid in list(range(1, self.num_seg_layers+1)):
                output = output + instance_text_features

            # reference_points: (bs, K, 16, 2)
            # reference_points_input: (bs, K, 16, 4, 2)
            reference_points = reference_points.detach()
            reference_points_input = reference_points[:, :, :, None] * src_valid_ratios[:, None, None]


            # Postitional Embedding Formulation
            query_pos = gen_point_pos_embed(reference_points_input[:, :, :, 0, :])  #  (bs, nq, 1, dim)
            query_pos = self.ref_point_head(query_pos)
            
            anchor_point = reference_points.mean(-2)
            query_pos_anchor = gen_point_pos_embed(anchor_point.unsqueeze(-2))  #  (bs, nq, 1, dim)
            query_pos_anchor = self.ref_point_head(query_pos_anchor)

            # Decoder Layer
            output = layer(
                output,                 # (bs, K, 16, 256)
                query_pos,              # (bs, K, 16, dim)
                query_pos_anchor,       # (bs, K, 1, dim)
                reference_points_input, # (bs, K, 16, 4, 2)
                src,                    # (bs, map_size, 256)
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
                lid
            )

            # Make Predictions
            if lid < self.num_seg_layers:
                reference_points, \
                segmentation_map, \
                lower_level_masks, \
                instance_text_features = \
                self.forward_prediction_reg_and_seg(output, fused_feature, query_pos, semantic_mask, \
                                                    reference_points, level_embed, lid)
            else:
                reference_points = self.forward_prediction_reg_only(output, reference_points, lid)
            output_class = self.forward_prediction_heads_class(output, query_pos, lid)
            
            

            if self.return_intermediate and lid < self.num_seg_layers:
                intermediate_reference_points.append(reference_points)
                intermediate_segmentation_map.append(segmentation_map)

            elif self.return_intermediate and lid >= self.num_seg_layers:
                intermediate_reference_points.append(reference_points)
                intermediate_segmentation_map.append(intermediate_segmentation_map[-1])

            intermediate_class.append(output_class)

        if self.return_intermediate:
            return torch.stack(intermediate_reference_points), \
                   torch.stack(intermediate_segmentation_map), \
                   torch.stack(intermediate_class), lower_level_masks

        return reference_points, segmentation_map, output_class


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")