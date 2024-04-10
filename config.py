"""This config is orginially from OpenMMLab: <link to github>"""

runai_run = False
output_dir = "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/" if not runai_run else "/storage/shoval/Anomaly_Detection_in_aerial_images/results/"
current_run_name = "weights_in_loss_and_sampling_bg_in_val_and_train_datasets"
anomaly_detector_cfg = dict(
    skip_stage=True,
    type="vit_based_anomaly_detector",
    vit_patch_size=8,
    vit_arch="vit_base",  # 'vit_tiny', 'vit_small', 'vit_base'
    vit_image_size=(512, 512),
    vit_threshold=None,
    pretrained_weights="",
    checkpoint_key="teacher",
    vit_model_mode="class_token_self_attention",  # "class_token_self_attention", "last_block_output"
    vit_model_type="dino_vit",  # "dino_vit", "dino_mc_vit"
    data_output_dir_name="sampled_extracted_bboxes_data",
    proposals_sizes=dict(square=(17,17), horizontal=(11,21), vertical=(21,11)),
    patches_filtering_threshold=85,

    patches_assigner=dict(
        type='mmdet.MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.1,
        min_pos_iou=0.5,
        match_low_quality=True,
        ignore_iof_thr=-1),

    train_dataloader=dict(
        batch_size=1,
        num_workers=2,
        # num_workers=0,
        persistent_workers=True,
        # persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='DOTAv2Dataset',
            data_root='/home/shoval/Documents/Repositories/data/gsd_normalized_dataset/train' if not runai_run else '/storage/shoval/vit_representation_experiment/gsd_normalized_dataset/train',
            ann_file='labelTxt/',
            data_prefix=dict(img_path='images/'),
            test_mode=True,
            pipeline=[
                dict(type='mmdet.LoadImageFromFile'),
                dict(
                    type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                dict(
                    type='ConvertBoxType',
                    box_type_mapping=dict(gt_bboxes='hbox')),
                dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
                dict(type='Normalize', mean=(0, 0, 0), std=(255.0, 255.0, 255.0), to_rgb=False),
                dict(type='Normalize', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
            ])),

    val_dataloader=dict(
        batch_size=1,
        num_workers=2,
        # num_workers=0,
        persistent_workers=True,
        # persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='DOTAv2Dataset',
            data_root='/home/shoval/Documents/Repositories/data/gsd_normalized_dataset/val' if not runai_run else '/storage/shoval/vit_representation_experiment/gsd_normalized_dataset/val',
            ann_file='labelTxt/',
            data_prefix=dict(img_path='images/'),
            test_mode=True,
            pipeline=[
                dict(type='mmdet.LoadImageFromFile'),
                dict(
                    type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                dict(
                    type='ConvertBoxType',
                    box_type_mapping=dict(gt_bboxes='hbox')),
                dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
                dict(type='Normalize', mean=(0, 0, 0), std=(255.0, 255.0, 255.0), to_rgb=False),
                dict(type='Normalize', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
            ])),

    test_dataloader = dict(
        batch_size=1,
        num_workers=2,
        # num_workers=0,
        persistent_workers=True,
        # persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='DOTAv2Dataset',
            data_root='/home/shoval/Documents/Repositories/data/gsd_normalized_dataset/test' if not runai_run else '/storage/shoval/vit_representation_experiment/gsd_normalized_dataset/test',
            ann_file='labelTxt/',
            data_prefix=dict(img_path='images/'),
            test_mode=True,
            pipeline=[
                dict(type='mmdet.LoadImageFromFile'),
                dict(
                    type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                dict(
                    type='ConvertBoxType',
                    box_type_mapping=dict(gt_bboxes='hbox')),
                dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
                dict(type='Normalize', mean=(0, 0, 0), std=(255.0, 255.0, 255.0), to_rgb=False),
                dict(type='Normalize', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
            ]))
)

classifier_cfg = dict(type="resnet18",
                      train_output_dir="train/Classifier",
                      test_output_dir="test/Classifier",
                      model_path='google/vit-base-patch16-224-in21k',
                      retrain=False,
                      resume=False,
                      max_epoch=100,
                      milestones=[30, 60, 90],
                      checkpoint_path="checkpoints",
                      train_batch_size=100,
                      val_batch_size=100,
                      dataloader_num_workers=10,
                      weighted_sampler=False,
                      loss_class_weights=True,
                      evaluate=False)

OOD_detector_cfg = dict(type="ODIN",
                        ood_class_names=["ship", "harbor", "roundabout", "helicopter", "swimming-pool", "storage-tank",
                                         "bridge"], save_outliers=False, num_of_outliers=50, rank_accord_features=True)
