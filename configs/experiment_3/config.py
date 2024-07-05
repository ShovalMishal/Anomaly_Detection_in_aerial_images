"""This config is originally from OpenMMLab: <link to github>"""
runai_run = False
output_dir = "/home/shoval/Documents/Repositories/Anomaly_Detection_in_aerial_images/results/" if not runai_run else "/storage/shoval/Anomaly_Detection_in_aerial_images/results/"
current_run_name = "experiment_3"
ood_class_names = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                   'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                   'basketball-court', 'soccer-ball-field', 'roundabout',
                   'swimming-pool', 'helicopter', 'container-crane', 'airport',
                   'helipad']

anomaly_detector_cfg = dict(
    train_dataloader=dict(
        batch_size=16,
        num_workers=16,
        # num_workers=0,
        persistent_workers=True,
        # persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='DOTAv2Dataset',
            data_root='/home/shoval/Documents/Repositories/data/gsd_08_normalized_dataset_rotated/train' if not runai_run else '/storage/shoval/datasets/gsd_normalized_dataset_rotated/train',
            ann_file='labelTxt/',
            data_prefix=dict(img_path='images/'),
            ood_labels=ood_class_names,
            ignore_ood_labels=False,
            pipeline=[
                dict(type='mmdet.LoadImageFromFile'),
                dict(
                    type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                dict(
                    type='ConvertBoxType',
                    # box_type_mapping=dict(gt_bboxes='hbox')),
                    box_type_mapping=dict(gt_bboxes='rbox')),
                dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
                dict(type='Normalize', mean=(0, 0, 0), std=(255.0, 255.0, 255.0), to_rgb=False),
                dict(type='Normalize', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
            ])),

    val_dataloader=dict(
        batch_size=16,
        num_workers=16,
        # num_workers=0,
        persistent_workers=True,
        # persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='DOTAv2Dataset',
            data_root='/home/shoval/Documents/Repositories/data/gsd_08_normalized_dataset_rotated/val' if not runai_run else '/storage/shoval/datasets/gsd_normalized_dataset_rotated/val',
            ann_file='labelTxt/',
            data_prefix=dict(img_path='images/'),
            ood_labels=ood_class_names,
            ignore_ood_labels=False,
            test_mode=True,
            pipeline=[
                dict(type='mmdet.LoadImageFromFile'),
                dict(
                    type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                dict(
                    type='ConvertBoxType',
                    box_type_mapping=dict(gt_bboxes='rbox')),
                dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
                dict(type='Normalize', mean=(0, 0, 0), std=(255.0, 255.0, 255.0), to_rgb=False),
                dict(type='Normalize', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
            ])),

    test_dataloader=dict(
        batch_size=16,
        num_workers=16,
        # num_workers=0,
        persistent_workers=True,
        # persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='DOTAv2Dataset',
            data_root='/home/shoval/Documents/Repositories/data/gsd_08_normalized_dataset_rotated/test' if not runai_run else '/storage/shoval/datasets/gsd_normalized_dataset_rotated/test',
            ann_file='labelTxt/',
            data_prefix=dict(img_path='images/'),
            ood_labels=ood_class_names,
            ignore_ood_labels=False,
            test_mode=True,
            pipeline=[
                dict(type='mmdet.LoadImageFromFile'),
                dict(
                    type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                dict(
                    type='ConvertBoxType',
                    box_type_mapping=dict(gt_bboxes='rbox')),
                dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
                dict(type='Normalize', mean=(0, 0, 0), std=(255.0, 255.0, 255.0), to_rgb=False),
                dict(type='Normalize', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
            ])),

    skip_stage=False,
    extract_patches=True,
    type="vit_based_anomaly_detector",
    vit_patch_size=8,
    vit_arch="vit_base",  # 'vit_tiny', 'vit_small', 'vit_base'
    vit_image_size=(512, 512),
    attention_num=9,
    vit_threshold=None,
    pretrained_weights="",
    checkpoint_key="teacher",
    vit_model_mode="class_token_self_attention",  # "class_token_self_attention", "last_block_output"

    vit_model_type="dino_vit",  # "dino_vit", "dino_mc_vit"
    data_output_dir_name="sampled_extracted_bboxes_data_regressor_ver",
    proposals_sizes=dict(square=(17, 17), horizontal=(11, 21), vertical=(21, 11)),
    patches_filtering_threshold=85,
    patches_assigner=dict(
        type='mmdet.MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.1,
        min_pos_iou=0.5,
        iou_calculator=dict(type='RBboxOverlaps2D'),
        match_low_quality=True,
        ignore_iof_thr=-1,
        _scope_="mmrotate"),

    bbox_regressor=dict(
        env_cfg=dict(
            cudnn_benchmark=False,
            mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
            dist_cfg=dict(backend='nccl')),
        val_evaluator=dict(type='BBoxRegressorMetric', scale_ranges=[(12.02, 24.04)]),
        test_evaluator=dict(type='BBoxRegressorMetric', scale_ranges=[(12.02, 24.04)]),
        train_dataloader=dict(
            batch_size=16,
            num_workers=16,
            persistent_workers=True,
            drop_last=False,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=dict(
                type='DOTAv2DatasetOOD3',
                data_root='/home/shoval/Documents/Repositories/data/gsd_08_normalized_dataset_rotated/train' if not runai_run else '/storage/shoval/datasets/gsd_normalized_dataset_rotated/train',
                ann_file='labelTxt/',
                data_prefix=dict(img_path='images/'),
                ood_labels=ood_class_names,
                ignore_ood_labels=True,
                extracted_patches_folder=None,
                pipeline=[
                    dict(type='mmdet.LoadImageFromFile'),
                    dict(
                        type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                    dict(
                        type='ConvertBoxType',
                        box_type_mapping=dict(gt_bboxes='rbox')),
                    # box_type_mapping=dict(gt_bboxes='hbox')),
                    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
                    dict(type='Normalize', mean=(0, 0, 0), std=(255.0, 255.0, 255.0), to_rgb=False),
                    dict(type='Normalize', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    dict(
                        type='mmdet.PackDetInputs',
                        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
                ])),
        val_dataloader=dict(
            batch_size=16,
            num_workers=16,
            persistent_workers=True,
            drop_last=False,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=dict(
                type='DOTAv2DatasetOOD3',
                data_root='/home/shoval/Documents/Repositories/data/gsd_08_normalized_dataset_rotated/val' if not runai_run else '/storage/shoval/datasets/gsd_normalized_dataset_rotated/val',
                ann_file='labelTxt/',
                data_prefix=dict(img_path='images/'),
                ood_labels=ood_class_names,
                ignore_ood_labels=True,
                extracted_patches_folder=None,
                test_mode=True,
                pipeline=[
                    dict(type='mmdet.LoadImageFromFile'),
                    dict(
                        type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                    dict(
                        type='ConvertBoxType',
                        box_type_mapping=dict(gt_bboxes='rbox')),
                    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
                    dict(type='Normalize', mean=(0, 0, 0), std=(255.0, 255.0, 255.0), to_rgb=False),
                    dict(type='Normalize', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    dict(
                        type='mmdet.PackDetInputs',
                        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
                ])),
        test_dataloader=dict(
            batch_size=16,
            num_workers=16,
            persistent_workers=True,
            drop_last=False,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=dict(
                type='DOTAv2DatasetOOD3',
                data_root='/home/shoval/Documents/Repositories/data/gsd_08_normalized_dataset_rotated/test' if not runai_run else '/storage/shoval/datasets/gsd_normalized_dataset_rotated/test',
                ann_file='labelTxt/',
                data_prefix=dict(img_path='images/'),
                ood_labels=ood_class_names,
                ignore_ood_labels=True,
                extracted_patches_folder=None,
                test_mode=True,
                pipeline=[
                    dict(type='mmdet.LoadImageFromFile'),
                    dict(
                        type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
                    dict(
                        type='ConvertBoxType',
                        box_type_mapping=dict(gt_bboxes='rbox')),
                    dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
                    dict(type='Normalize', mean=(0, 0, 0), std=(255.0, 255.0, 255.0), to_rgb=False),
                    dict(type='Normalize', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    dict(
                        type='mmdet.PackDetInputs',
                        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
                ])),
        train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=1),
        val_cfg=dict(type='ValLoop'),
        test_cfg=dict(type='TestLoop'),
        default_scope='mmrotate',
        default_hooks=dict(
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook', interval=50),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1),
            sampler_seed=dict(type='DistSamplerSeedHook'),
            visualization=dict(type='mmdet.DetVisualizationHook'),
            # early_stopping=dict(
            #     type="EarlyStoppingHook",
            #     monitor="dota/recall",
            #     strict=True,
            #     patience=5,
            #     rule='greater')
        ),
        param_scheduler=[
            dict(
                type='LinearLR',
                start_factor=0.3333333333333333,
                by_epoch=True,
                begin=0,
                end=1),
            dict(
                type='MultiStepLR',
                begin=0,
                end=15,
                by_epoch=True,
                milestones=[11, 14],
                gamma=0.1)
        ],
        # optim_wrapper=dict(
        #     type='OptimWrapper',
        #     optimizer=dict(type='AdamW', lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001),
        #     clip_grad=dict(max_norm=10, norm_type=2)
        # ),
        optim_wrapper=dict(
            type='OptimWrapper',
            optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
            clip_grad=dict(max_norm=10, norm_type=2)),
        visualizer=dict(
            type='RotLocalVisualizer',
            vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
            name='visualizer'),
        log_processor=dict(type='LogProcessor', window_size=50, by_epoch=True),
        log_level='INFO',
        load_from=None,
        resume=True,

        model=dict(
            type='mmdet.BBoxRegressor',
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            angle_version='le90',
            bbox_roi_extractor=dict(
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=7,
                    sample_num=2,
                    clockwise=True),
                out_channels=768,
                featmap_strides=[8]),
            bbox_head=dict(
                type='mmdet.Shared2FCBBoxHead',
                predict_box_type='rbox',
                in_channels=768,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=2,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                bbox_coder=dict(
                    type='DeltaXYWHTRBBoxCoder',
                    angle_version='le90',
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                    target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
                reg_class_agnostic=True,
                with_cls=False,
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=0.0),
                loss_bbox=dict(
                    type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
            test_cfg=dict(bbox_regressor_mode=True),
        )
    ),
)

classifier_cfg = dict(type="resnet18",
                      train_output_dir="train/Classifier",
                      test_output_dir="test/Classifier",
                      model_path='google/vit-base-patch16-224-in21k',
                      retrain=True,
                      resume=False,
                      max_epoch=100,
                      milestones=[30, 60, 90],
                      checkpoint_path="checkpoints",
                      train_batch_size=100,
                      val_batch_size=100,
                      dataloader_num_workers=10,
                      weighted_sampler=False,
                      loss_class_weights=True,
                      evaluate=True)

OOD_detector_cfg = dict(type="ODIN",
                        ood_class_names=ood_class_names, save_outliers=True, num_of_outliers=50,
                        rank_accord_features=True)
