"""
Implements the knowledge distillation loss
"""
from abc import get_cache_token
import torch
from torch.nn import functional as F
from torch.nn.modules.loss import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from utils import batch_index_select
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import math

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class DiffPruningLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, dynamic=False, ratio_weight=0.5, pruning_loc=[3,6,9], keep_ratio=[0.75, 0.5, 0.25], clf_weight=0, print_mode=True):
        super().__init__()
        self.base_criterion = base_criterion
        self.pruning_loc = pruning_loc
        self.keep_ratio = keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.cut_loss = 0
        
        
        self.cls_weight = clf_weight
        self.ratio_weight = ratio_weight
        self.cut_weight = 10.0


    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        pred, out_attns, out_attn_masks, out_features = outputs  # out_attns: B,N,1    out_features: B,N,C
        
        # cut loss
        cut_loss = 0.0
        for i, mask in enumerate(out_attn_masks):
            if i < len(out_attn_masks)-2:
                W = out_attns[i].softmax(-1) # B,H,N,N
                B, H, N, _ = W.shape
                samecut = mask.reshape(B,N,1) * mask.reshape(B,1,N) # B,N,N
                intra = W * samecut.reshape(B,1,N,N)
                intra_loss = F.mse_loss(intra.sum(-1), torch.ones(B,H,N, device=intra.device) * mask.detach().reshape(B,1,N))
                cut_loss = cut_loss + intra_loss
            else:
                W = out_attns[i].softmax(-1) # B,H,N,N
                B, H, N, _ = W.shape
                samecut = mask.reshape(B,N,1) * mask.reshape(B,1,N) # B,N,N
                samecut[:,1:,0] = 0.0
                intra = W * samecut.reshape(B,1,N,N)
                intra_loss = F.mse_loss(intra.sum(-1), torch.ones(B,H,N, device=intra.device) * mask.detach().reshape(B,1,N))
                cut_loss = cut_loss + intra_loss
        
        # classification loss
        cls_loss = self.base_criterion(pred, labels)
        
        # print(cls_loss, pred_loss, cut_loss)
        loss = self.cls_weight * cls_loss + self.cut_weight * cut_loss / len(self.pruning_loc)
        
        if self.print_mode:
            self.cls_loss += cls_loss.item()
            self.cut_loss += cut_loss.item()
            self.count += 1
            if self.count == 100:
                print('loss info: cls_loss=%.4f, cut_loss=%.4f' % (self.cls_loss / 100, self.cut_loss / 100))
                self.count = 0
                self.cls_loss = 0
                self.cut_loss = 0
        return loss


class DistillDiffPruningLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, base_criterion: torch.nn.Module, ratio_weight=2.0, distill_weight=0.5, dynamic=False, pruning_loc=[3,6,9], keep_ratio=[0.75, 0.5, 0.25], clf_weight=0, mse_token=False, print_mode=True):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.pruning_loc = pruning_loc
        self.keep_ratio = keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.cls_distill_loss = 0
        self.token_distill_loss = 0
        self.mse_token = mse_token
        self.dynamic = dynamic

        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight
        
        self.cut_loss = 0
        self.cut_weight = 10.0
        self.recover_loss = 0
        self.recover_weight = 10000.0

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        cls_s, token_s, out_attns, out_attn_masks, out_xs = outputs
        with torch.no_grad():
            cls_t, token_t, out_cls = self.teacher_model(inputs)
        
        # cut loss
        cut_loss = 0.0
        """
        for i, mask in enumerate(out_attn_masks):
            W = out_attns[i].softmax(-1) # B,H,N,N
            B, H, N, _ = W.shape
            samecut = mask.reshape(B,N,1) * mask.reshape(B,1,N) # B,N,N
            intra = W * samecut.reshape(B,1,N,N)
            intra_loss = F.mse_loss(intra.sum(-1), torch.ones(B,H,N, device=intra.device) * mask.detach().reshape(B,1,N))
            cut_loss = cut_loss + intra_loss
        """
        
        # recover loss
        recover_loss = 0.0
        for i in range(len(out_xs)):
            recover_loss = recover_loss + F.mse_loss(F.normalize(out_xs[i][0]), F.normalize(out_xs[i][1]))
        
        # classification loss
        cls_loss = self.base_criterion(cls_s, labels)
        
        # distilled classification loss
        cls_kl_loss = F.kl_div(
                F.log_softmax(cls_s, dim=-1),
                F.log_softmax(cls_t, dim=-1),
                reduction='batchmean',
                log_target=True
            )
        
        # distilled feature loss
        token_kl_loss = 0.0
        if len(token_s.shape) == 2:
            token_kl_loss = token_kl_loss + F.mse_loss(F.normalize(token_s), F.normalize(token_t))
        else:
            token_kl_loss = token_kl_loss + F.kl_div(F.log_softmax(token_s * out_attn_masks[-1][:,1:,:], dim=-1),
                                                     F.log_softmax(token_t * out_attn_masks[-1][:,1:,:], dim=-1),
                                                     reduction='batchmean',
                                                     log_target=True)
        
        # print(cls_loss, pred_loss)
        loss = self.clf_weight * cls_loss + self.cut_weight * cut_loss / len(self.pruning_loc) + self.distill_weight * cls_kl_loss + self.distill_weight * 100 * token_kl_loss + self.recover_weight * recover_loss / len(self.pruning_loc)

        if self.print_mode:
            self.cls_loss += cls_loss.item()
            self.cls_distill_loss += cls_kl_loss.item()
            self.token_distill_loss += token_kl_loss.item()
            #self.cut_loss += cut_loss.item()
            self.recover_loss += recover_loss.item()
            self.count += 1
            if self.count == 100:
                print('loss info: cls_loss=%.4f, cls_kl=%.4f, token_kl=%.4f, cut_loss=%.4f, recover_loss=%.4f' % (self.cls_loss/100, self.cls_distill_loss/100, self.token_distill_loss/100, self.cut_loss/100, self.recover_loss/100))
                self.count = 0
                self.cls_loss = 0
                self.cls_distill_loss = 0
                self.token_distill_loss = 0
                self.cut_loss = 0
                self.recover_loss = 0
        return loss

