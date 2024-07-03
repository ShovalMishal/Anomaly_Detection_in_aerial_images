import time

from bg_subtractor_utils import extract_patches_accord_heatmap
from mmdet.utils import OptMultiConfig
import os
from torch import Tensor
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import InstanceList
from mmrotate.structures.bbox import (RotatedBoxes, hbox2rbox)
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox2roi
from mmengine.model import BaseModel
from mmdet.structures import OptSampleList, SampleList
from mmengine.config import ConfigDict
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import unpack_gt_instances, samplelist_boxtype2tensor
from mmdet.registry import MODELS, TASK_UTILS
import torch
import numpy as np
import random
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.set_seed(0)

# Set a random seed for PyTorch
seed = 42
torch.manual_seed(seed)

# Set a random seed for NumPy (if you use NumPy alongside PyTorch)
np.random.seed(seed)

# Set a random seed for Python's built-in random module (if needed)
random.seed(seed)
# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    # Set a random seed for the GPU
    torch.cuda.manual_seed_all(seed)


@MODELS.register_module()
class BBoxRegressor(BaseModel):
    def __init__(self, assigner, angle_version, bbox_roi_extractor, bbox_head, test_cfg,
                 init_cfg: OptMultiConfig = None, ):
        super().__init__(init_cfg=init_cfg)
        self.proposals_cache_path = None
        self.dino_vit_bg_subtractor = None
        self.patches_filtering_threshold = None
        self.proposal_sizes = None
        self.features_extractor = None
        self.logger = None
        self.vit_patch_size = None
        self.output_dir = None
        self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
        self.bbox_head = MODELS.build(bbox_head)
        self.angle_version = angle_version
        self.assigner = TASK_UTILS.build(assigner)
        self.test_cfg = test_cfg

    def initialize(self, logger, vit_patch_size, features_extractor, dino_vit_bg_subtractor,
                   proposal_sizes, patches_filtering_threshold, output_dir):
        self.logger = logger
        self.vit_patch_size = vit_patch_size
        self.features_extractor = features_extractor
        self.proposal_sizes = proposal_sizes
        self.patches_filtering_threshold = patches_filtering_threshold
        self.dino_vit_bg_subtractor = dino_vit_bg_subtractor
        self.output_dir = output_dir
        self.proposals_cache_path = os.path.join(self.output_dir, "cache_folder")
        os.makedirs(self.proposals_cache_path, exist_ok=True)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale=True):
        x, proposals_list = self.extract_feat_and_proposals(batch_inputs, batch_data_samples)
        assert len(proposals_list) == len(batch_data_samples)

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        proposals = [res.priors for res in proposals_list]
        rois = bbox2roi(proposals)
        bbox_results = self.bbox_forward(x, rois)
        # bbox_loss_and_target, _ = self.calculate_loss(x=x, proposals_list=proposals_list,
        #                                                          batch_data_samples=batch_data_samples,
        #                                                          add_gt_as_proposals=False)
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        if isinstance(bbox_preds, torch.Tensor):
            bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
        else:
            bbox_preds = self.bbox_head.bbox_pred_split(
                bbox_preds, num_proposals_per_img)
        results_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=tuple([None] * len(bbox_preds)),
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        # for data_samples in batch_data_samples:
        #     data_samples.loss_bbox = bbox_loss_and_target['bbox_targets'][2].shape[0] * bbox_loss_and_target['loss_bbox']['loss_bbox']
        #     data_samples.num_of_regressions = bbox_loss_and_target['bbox_targets'][2].shape[0]
        return batch_data_samples

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: InstanceList) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples

    def bbox_forward(self, x, rois):
        bbox_feats = self.bbox_roi_extractor(
            x, rois)
        _, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def extract_feat_and_proposals(self, batch_inputs: Tensor, batch_data_samples):
        proposals_list = []
        all_features = []
        for img, batch_data_sample in zip(batch_inputs, batch_data_samples):
            w_featmap = img.shape[-2] // self.vit_patch_size
            h_featmap = img.shape[-1] // self.vit_patch_size
            with torch.no_grad():
                features = self.features_extractor.get_last_block(img.unsqueeze(0))[1:, :]
            features = (features.permute(1, 0).reshape(-1, w_featmap, h_featmap).unsqueeze(dim=0))
            predicted_patches = batch_data_sample.predicted_patches.predicted_patches
            all_features.append(features)

            proposals_bboxes = RotatedBoxes(
                RotatedBoxes(hbox2rbox(predicted_patches)).regularize_boxes(self.angle_version))
            results = InstanceData()
            results.priors = proposals_bboxes
            proposals_list.append(results)
            # del batch_data_sample.predicted_patches

        all_features = torch.cat(all_features, dim=0)
        x = (all_features,)
        return x, proposals_list

    def calculate_loss(self, x, proposals_list, batch_data_samples: SampleList, add_gt_as_proposals, concat=True):
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs
        num_imgs = len(batch_data_samples)
        sampling_results = []
        # assign gts to proposals
        for i in range(num_imgs):
            assign_result = self.assigner.assign(
                proposals_list[i], batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            gt_bboxes = batch_gt_instances[i].bboxes
            priors = proposals_list[i].priors
            gt_labels = batch_gt_instances[i].labels
            gt_flags = priors.new_zeros((priors.shape[0],), dtype=torch.uint8)
            if add_gt_as_proposals and len(gt_bboxes) > 0:
                priors = cat_boxes([gt_bboxes, priors], dim=0)
                assign_result.add_gt_(gt_labels)
                gt_ones = priors.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
                gt_flags = torch.cat([gt_ones, gt_flags])
            pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            if pos_inds.numel() != 0:
                pos_inds = pos_inds.squeeze(1).unique()
            sampling_result = SamplingResult(pos_inds=pos_inds, neg_inds=pos_inds.new_empty(size=(0, 1)), priors=priors,
                                             gt_bboxes=gt_bboxes, assign_result=assign_result, gt_flags=gt_flags)
            sampling_results.append(sampling_result)
        # first element turn to be img_ind
        rois = bbox2roi([res.priors for res in sampling_results]).to(device)
        bbox_results = self.bbox_forward(x, rois)
        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=torch.tensor([]),
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=ConfigDict({'pos_weight': -1}),
            concat=concat)
        return bbox_loss_and_target, bbox_results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList, add_gt_as_proposals: bool = True):
        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("model_inference"):
        # time1 = time.time()
        x, proposals_list = self.extract_feat_and_proposals(batch_inputs, batch_data_samples)
        # time2 = time.time()
        # print(f"Time to extract features and proposals: {time2 - time1}")
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        assert len(proposals_list) == len(batch_data_samples)
        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("model_inference"):
        # time1 = time.time()
        bbox_loss_and_target, bbox_results = self.calculate_loss(x=x, proposals_list=proposals_list,
                                                                 batch_data_samples=batch_data_samples,
                                                                 add_gt_as_proposals=add_gt_as_proposals)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        bbox_results.update(bbox_loss_and_target['loss_bbox'])
        # time2 = time.time()
        # print(f"Time to calculate loss: {time2 - time1}")
        return bbox_results

