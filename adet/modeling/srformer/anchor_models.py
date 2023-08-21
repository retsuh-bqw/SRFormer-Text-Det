import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from adet.layers.anchor_deformable_transformer import DeformableTransformer_Det
from adet.utils.misc import NestedTensor, inverse_sigmoid_offset, nested_tensor_from_tensor_list, sigmoid_offset
from .utils import MLP, inverse_sigmoid, CirConv_score
import fvcore.nn.weight_init as weight_init




class SRFormer(nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = backbone

        self.d_model = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.num_proposals = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.pos_embed_scale = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_ctrl_points = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        self.num_classes = 1  # only text
        self.sigmoid_offset = not cfg.MODEL.TRANSFORMER.USE_POLYGON

        self.ctrl_point_embed = nn.Embedding(16, self.d_model)
        self.num_seg_layers = cfg.MODEL.TRANSFORMER.SEG_LAYERS

        self.transformer = DeformableTransformer_Det(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels,
            dec_n_points=self.dec_n_points,
            enc_n_points=self.enc_n_points,
            num_proposals=self.num_proposals,
            num_ctrl_points=self.num_ctrl_points,
            num_seg_layers=self.num_seg_layers
        )
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)

        self.bbox_coord = MLP(self.d_model, self.d_model, 4, 3)
        self.bbox_class = nn.Linear(self.d_model, self.num_classes)

        self.mask_embed = MLP(self.d_model, self.d_model, self.d_model, 2)
        self.offset_embed = MLP(self.d_model, self.d_model, 2, 3)
        self.anchor_offset_embed = MLP(self.d_model, self.d_model, 2, 3)

        self.query_aggregation_weights = nn.Sequential( CirConv_score(self.d_model),
                                                        nn.Linear(self.d_model, 1, bias=False),
                                                        nn.Sigmoid())

        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            num_channels = [512, 1024, 2048]
            num_backbone_outs = len(strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model,kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                )
            ])
        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.anchor_offset_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.anchor_offset_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.offset_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.offset_embed.layers[-1].bias.data, 0)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.num_decoder_layers
        
        self.mask_feature_proj = nn.Conv2d(
            self.d_model,
            self.d_model,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_feature_proj)
        self.ctrl_point_class = nn.ModuleList([self.ctrl_point_class for _ in range(num_pred)])
        self.offset_embed = nn.ModuleList([self.offset_embed for _ in range(num_pred)])
        self.anchor_offset_embed = nn.ModuleList([self.anchor_offset_embed for _ in range(self.num_seg_layers)])
        self.mask_embed = nn.ModuleList([self.mask_embed for _ in range(self.num_seg_layers)])

        self.transformer.decoder_new.bbox_embed = None
        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_embed = self.bbox_coord
        self.decoder_norm = nn.LayerNorm(self.d_model)

        self.transformer.decoder_new.decoder_norm = self.decoder_norm
        self.transformer.decoder_new.mask_embed = self.mask_embed
        self.transformer.decoder_new.device = self.device
        self.transformer.decoder_new.mask_feature_proj = self.mask_feature_proj
        self.transformer.decoder_new.offset_embed = self.offset_embed
        self.transformer.decoder_new.anchor_offset_embed = self.anchor_offset_embed
        self.transformer.decoder_new.query_aggregation_weights = self.query_aggregation_weights
        self.transformer.decoder_new.ctrl_point_class = self.ctrl_point_class
        self.transformer.decoder_new.ctrl_point_embed = self.ctrl_point_embed

        self.to(self.device)


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)


        input_shape = samples.tensor.shape
        features, pos = self.backbone(samples)
        
        bk_features = [features[i].decompose()[0] for i in range(4)]
        features = features[1:]
        
        if self.num_feature_levels == 1:
            raise NotImplementedError

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks[1]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        
        # Ignore one-forth faeture pos
        pos = pos[1:]

        ctrl_point_embed = self.ctrl_point_embed.weight[None,:,:].repeat(self.num_proposals,1,1)

        outputs_coord, enc_outputs_class, enc_outputs_coord_unact, \
        outputs_seg_mask, semantic_seg_mask, outputs_class, lower_level_masks = self.transformer(
            srcs, masks, pos, ctrl_point_embed, bk_features
        )

        out = {'pred_logits': outputs_class[-1], 'pred_ctrl_points': outputs_coord[-1], 'pred_seg_mask':outputs_seg_mask[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_seg_mask)

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        out['semantic_seg_mask'] = semantic_seg_mask

        out['lower_level_masks'] = lower_level_masks

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_seg_mask):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {'pred_logits': a, 'pred_ctrl_points': b, 'pred_seg_mask': c} \
            for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_seg_mask[:-1])
        ]