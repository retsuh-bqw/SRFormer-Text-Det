"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from adet.utils.misc import box_cxcywh_to_xyxy, generalized_box_iou
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import point_sample
from torch.cuda.amp import autocast

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor
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
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def sigmoid_ce_loss(
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

    return loss.mean(-1)

class BoxHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            giou_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        """Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            giou_weight: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.giou_weight = giou_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0 or giou_weight != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            
            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])


            # Compute the classification cost.
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - \
                neg_cost_class

            
            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox)
            )

            # Final cost matrix
            C = self.coord_weight * cost_bbox + self.class_weight * \
                cost_class + self.giou_weight * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(
                c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class CtrlPointHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            num_seg_layers: int = 2
    ):
        """Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: This is the relative weight of the L1 error of the keypoint coordinates in the matching cost
        """
        super().__init__()
        self.class_weight = 2
        self.coord_weight = 5
        # self.anchor_weight = coord_weight
        self.mask_weight = 5
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.num_points = 12544
        self.num_seg_layers = num_seg_layers
        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"

    def forward(self, outputs, targets, lvl):
        with torch.no_grad():
            coord_weight = self.coord_weight
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            # [batch_size, n_queries, n_points, 2] --> [batch_size * num_queries, n_points * 2]
            out_pts = outputs["pred_ctrl_points"].flatten(0, 1).flatten(-2)
            tgt_pts = torch.cat([v["ctrl_points"] for v in targets])

            cost_mask_dice = 0
            cost_mask_ce = 0
            if lvl < self.num_seg_layers:
                # coord_weight /= 4
                out_masks = outputs['pred_seg_mask'].flatten(0, 1)
                # out_anchor_points = outputs['pred_anchor_points'].flatten(0, 1)
                tgt_masks = torch.cat([v["segmentation_map"] for v in targets])
                tgt_masks = F.interpolate(tgt_masks.unsqueeze(1),size=out_masks.shape[-2:], mode="nearest")

                num_gt = tgt_masks.shape[0]
                out_masks = out_masks[:, None]



                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(1, self.num_points, 2, device=out_masks.device)
                # get gt labels
                tgt_masks = point_sample(
                    tgt_masks,
                    point_coords.repeat(tgt_masks.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_masks = point_sample(
                    out_masks,
                    point_coords.repeat(out_masks.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with autocast(enabled=False):
                    out_masks = out_masks.float()
                    tgt_masks = tgt_masks.float()

                    cost_mask_dice = dice_loss(out_masks.unsqueeze(1), tgt_masks.unsqueeze(0))
                    # cost_mask_ce = sigmoid_ce_loss(out_masks.unsqueeze(1).repeat(1,num_gt,1), tgt_masks.unsqueeze(0).repeat(out_masks.shape[0], 1, 1))


            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                             (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * \
                             ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            # hack here for label ID 0
            cost_class = pos_cost_class - neg_cost_class
            cost_kpts = torch.cdist(out_pts, tgt_pts.flatten(-2), p=1) #/ 16

            C = self.class_weight * cost_class + coord_weight * cost_kpts + \
                self.mask_weight * cost_mask_dice + self.mask_weight * cost_mask_ce

            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["ctrl_points"]) for v in targets]
            indices = [linear_sum_assignment(
                c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    num_seg_layers = cfg.MODEL.TRANSFORMER.SEG_LAYERS
    cfg = cfg.MODEL.TRANSFORMER.LOSS

    return BoxHungarianMatcher(class_weight=cfg.BOX_CLASS_WEIGHT,
                               coord_weight=cfg.BOX_COORD_WEIGHT,
                               giou_weight=cfg.BOX_GIOU_WEIGHT,
                               focal_alpha=cfg.FOCAL_ALPHA,
                               focal_gamma=cfg.FOCAL_GAMMA), \
        CtrlPointHungarianMatcher(class_weight=cfg.POINT_CLASS_WEIGHT,
                                 coord_weight=cfg.POINT_COORD_WEIGHT,
                                 focal_alpha=cfg.FOCAL_ALPHA,
                                 focal_gamma=cfg.FOCAL_GAMMA,
                                 num_seg_layers=num_seg_layers)