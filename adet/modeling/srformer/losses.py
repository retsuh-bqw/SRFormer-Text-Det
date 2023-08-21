import torch
import torch.nn as nn
import torch.nn.functional as F
from adet.utils.misc import accuracy, generalized_box_iou, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, is_dist_avail_and_initialized
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def sigmoid_focal_loss(inputs, targets, num_inst, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss.ndim == 4:
        return loss.mean((1, 2)).sum() / num_inst
    elif loss.ndim == 3:
        return loss.mean(1).sum() / num_inst
    else:
        raise NotImplementedError(f"Unsupported dim {loss.ndim}")


class SetCriterion(nn.Module):
    def __init__(
            self,
            num_classes,
            enc_matcher,
            dec_matcher,
            weight_dict,
            enc_losses,
            dec_losses,
            num_ctrl_points,
            focal_alpha=0.25,
            focal_gamma=2.0,
            num_seg_layers=2
    ):
        """ Create the criterion.
        Parameters:
            - num_classes: number of object categories, omitting the special no-object category
            - matcher: module able to compute a matching between targets and proposals
            - weight_dict: dict containing as key the names of the losses and as values their relative weight.
            - losses: list of all the losses to be applied. See get_loss for list of available losses.
            - focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.enc_matcher = enc_matcher
        self.dec_matcher = dec_matcher
        self.weight_dict = weight_dict
        self.enc_losses = enc_losses
        self.dec_losses_all = dec_losses + ['instance_seg']
        self.dec_losses_reg = dec_losses
        # print(dec_losses)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_ctrl_points = num_ctrl_points
        self.mask_weight = 5
        self.reg_weight = 5
        self.class_weight = 2
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75
        self.num_points = 12544
        self.num_seg_layers = num_seg_layers

    def loss_labels(self, outputs, targets, indices, num_inst, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(
            src_logits.shape[:-1], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        if len(target_classes_o.shape) < len(target_classes[idx].shape):
            target_classes_o = target_classes_o[..., None]
        target_classes[idx] = target_classes_o

        shape = list(src_logits.shape)
        shape[-1] += 1
        target_classes_onehot = torch.zeros(
            shape, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device
        )
        target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[..., :-1]
        # src_logits, target_classes_onehot: (bs, nq, n_pts, 1)
        loss_ce = self.class_weight * sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_inst, alpha=self.focal_alpha, gamma=self.focal_gamma
        ) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses
    
    def sigmoid_ce_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        ):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor
        """
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # inputs = ((inputs - 0.5)*5).sigmoid()
        # loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        # print(loss.shape)

        return loss.mean(-1).sum()

    def dice_loss(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            num_masks: float,
        ):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()

        # print(inputs.max(), inputs.min())
        # inputs = inputs.flatten(1)
        # targets = targets.flatten(1)
        # print(inputs.shape, targets.shape)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_inst):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.mean(-2).argmax(-1) == 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_inst):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_inst

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses['loss_giou'] = loss_giou.sum() / num_inst
        return losses

    def loss_semantic_seg(self, outputs, targets, indices, num_inst):
        target_segmentation_masks = torch.stack([v["segmentation_map"].sum(0) for v in targets])
        # for tgt_mask in target_segmentation_masks:
        # src_segmentation_masks = outputs['pred_seg_mask']
        # src_segmentation_masks = outputs['pred_seg_mask'].sum(dim=1).unsqueeze(1)
        # print(target_segmentation_masks.shape)
        semantic_seg_mask = outputs['semantic_seg_mask']
        target_segmentation_masks = F.interpolate(target_segmentation_masks.unsqueeze(1),size=semantic_seg_mask.shape[-2:], mode="nearest").squeeze(dim=1)
        # target_segmentation_masks = target_segmentation_masks.sum(dim=0).flatten()
        # print(target_segmentation_masks.max())
        target_segmentation_masks = torch.clamp(target_segmentation_masks, 0, 1)
        # print(target_segmentation_masks.shape, semantic_seg_mask.shape)
        b, _, h, w = semantic_seg_mask.shape
        # print(semantic_seg_mask.shape, src_segmentation_masks.shape)
        # print(target_segmentation_masks.shape, semantic_seg_mask.shape)
        loss_dice = 0
        loss_ce = 0
        # loss_ce_constraint = 0
        # loss_dice_constraint = 0
        for i in range(b):
            loss_dice += self.dice_loss(semantic_seg_mask[i].flatten().unsqueeze(0), target_segmentation_masks[i].flatten().unsqueeze(0), 1) * self.mask_weight
            loss_ce += self.sigmoid_ce_loss(semantic_seg_mask[i].flatten().unsqueeze(0), target_segmentation_masks[i].flatten().unsqueeze(0)) * self.mask_weight

            # loss_dice_constraint += self.dice_loss(src_segmentation_masks[i].flatten().unsqueeze(0), target_segmentation_masks[i].flatten().unsqueeze(0), 1) #* self.mask_weight
            # loss_ce_constraint += self.sigmoid_ce_loss(src_segmentation_masks[i].flatten().unsqueeze(0), target_segmentation_masks[i].flatten().unsqueeze(0)) #* self.mask_weight
        loss_dice /= b
        loss_ce /= b
        # loss_dice_constraint /= b
        # loss_ce_constraint /= b
        losses = {'loss_semantic_seg_dice':loss_dice, 'loss_semantic_seg_ce':loss_ce,}
                #   'loss_seg_dice_constraint':loss_dice_constraint, 'loss_seg_ce_constraint':loss_ce_constraint}
        return losses

    def loss_lower_level_mask(self, outputs, targets, indices, num_inst):
        idx = self._get_src_permutation_idx(indices)
        loss_dice = 0
        loss_mask_ce = 0
        for lvl in range(3):
            src_segmentation_masks = outputs['lower_level_masks'][lvl][idx]
            target_segmentation_masks = torch.cat([t['segmentation_map'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            target_segmentation_masks = F.interpolate(target_segmentation_masks.unsqueeze(1),size=src_segmentation_masks.shape[-2:], mode="nearest").squeeze()

            if len(target_segmentation_masks.shape) == 2:
                target_segmentation_masks = target_segmentation_masks.unsqueeze(0)

            with torch.no_grad():
                # sample point_coords
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_segmentation_masks[:,None],
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points // (4 ** (lvl+1)),
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # get gt labels
                point_labels = point_sample(
                    target_segmentation_masks[:,None],
                    point_coords,
                    align_corners=False,
                ).squeeze(1)

            point_logits = point_sample(
                src_segmentation_masks[:,None],
                point_coords,
                align_corners=False,
            ).squeeze(1)


            loss_dice += self.dice_loss(point_logits, point_labels, 1) * self.mask_weight
            loss_mask_ce += self.sigmoid_ce_loss(point_logits, point_labels) * self.mask_weight
        
        loss_dice /= 3
        loss_mask_ce /= 3
        
        losses = {'loss_lower_instance_mask_dice':loss_dice / num_inst,\
                  'loss_lower_instance_mask_ce':loss_mask_ce / num_inst}

        return losses


    def loss_instance_seg_mask(self, outputs, targets, indices, num_inst):
        idx = self._get_src_permutation_idx(indices)
        # src_anchor_points = outputs['pred_anchor_points'][idx]
        src_segmentation_masks = outputs['pred_seg_mask'][idx]

        target_segmentation_masks = torch.cat([t['segmentation_map'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_segmentation_masks = F.interpolate(target_segmentation_masks.unsqueeze(1),size=src_segmentation_masks.shape[-2:], mode="nearest").squeeze()

        if len(target_segmentation_masks.shape) == 2:
            target_segmentation_masks = target_segmentation_masks.unsqueeze(0)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_segmentation_masks[:,None],
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_segmentation_masks[:,None],
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_segmentation_masks[:,None],
            point_coords,
            align_corners=False,
        ).squeeze(1)


        loss_dice = self.dice_loss(point_logits, point_labels, 1) * self.mask_weight
        loss_mask_ce = self.sigmoid_ce_loss(point_logits, point_labels) * self.mask_weight

        losses = {'loss_instance_mask_dice':loss_dice / num_inst,\
                  'loss_instance_mask_ce':loss_mask_ce / num_inst}

        return losses


    def loss_ctrl_points(self, outputs, targets, indices, num_inst):
        """Compute the losses related to the keypoint coordinates, the L1 regression loss
        """
        assert 'pred_ctrl_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ctrl_points = outputs['pred_ctrl_points'][idx]
        target_ctrl_points = torch.cat([t['ctrl_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    
        loss_ctrl_points = self.reg_weight * F.l1_loss(src_ctrl_points, target_ctrl_points, reduction='sum') #/ src_ctrl_points.shape[1]

        losses = {'loss_ctrl_points': loss_ctrl_points / num_inst}
        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_inst,  **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'ctrl_points': self.loss_ctrl_points,
            'boxes': self.loss_boxes,
            'instance_seg': self.loss_instance_seg_mask
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_inst, **kwargs)

    def forward(self, outputs, targets, gt_instances):
        """ This performs the loss computation.
        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                  The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.dec_matcher(outputs_without_aux, targets, lvl=5)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_inst = sum(len(t['ctrl_points']) for t in targets)
        num_inst = torch.as_tensor([num_inst], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_inst)
        num_inst = torch.clamp(num_inst / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.dec_losses_reg:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_inst,  **kwargs))
        losses.update(self.loss_semantic_seg(outputs, targets, indices, num_inst, **kwargs))
        losses.update(self.loss_lower_level_mask(outputs, targets, indices, num_inst, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.dec_matcher(aux_outputs, targets, i)
                losses_key = self.dec_losses_all if i < self.num_seg_layers else self.dec_losses_reg
                for loss in losses_key:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_inst, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.enc_matcher(enc_outputs, targets)
            for loss in self.enc_losses:
                kwargs = {}
                if loss == 'labels':
                    kwargs['log'] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, targets, indices, num_inst, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses