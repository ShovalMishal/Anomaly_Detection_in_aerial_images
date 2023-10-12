"""This config is orginially from OpenMMLab: <link to github>"""
dataset_type = 'DOTAv2Dataset'
data_root = './data/split_ss_dota/'
patches_dataset = dict(
    path="./data/patches_dataset/",
    shuffle=False,
    num_workers=1,
    inclusion_file_name="nonbg_dataset.txt",
    ood_classes=["large-vehicle"]  # "helicopter", "tennis-court",
)
patches_assigner = dict(
    type='mmdet.MaxIoUAssigner',
    pos_iou_thr=0.5,
    neg_iou_thr=0.1,
    min_pos_iou=0.5,
    match_low_quality=True,
    ignore_iof_thr=-1,
    iou_calculator=dict(type='RBbox2HBboxOverlaps2D', _scope_='mmrotate'))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=20,
    num_workers=2,
    # num_workers=0,
    persistent_workers=True,
    # persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTAv2Dataset',
        data_root='./data/split_ss_dota/',
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
            dict(
                type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(
                type='ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='rbox')),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    # persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTAv2Dataset',
        data_root='./data/split_ss_dota/',
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
            dict(
                type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(
                type='ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='rbox')),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))

subval_dataloader = dict(
    batch_size=1,
    num_workers=2,
    # num_workers=0,
    persistent_workers=True,
    # persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTAv2Dataset',
        data_root='./data/split_ss_dota/',
        ann_file='subval/annfiles/',
        data_prefix=dict(img_path='subval/images/'),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
            dict(
                type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(
                type='ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='rbox')),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))

subtest_dataloader = dict(
    batch_size=1,
    num_workers=2,
    # num_workers=0,
    persistent_workers=True,
    # persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTAv2Dataset',
        data_root='./data/split_ss_dota/',
        ann_file='subtest/annfiles/',
        data_prefix=dict(img_path='subtest/images/'),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
            dict(
                type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(
                type='ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='rbox')),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))

subtrain_dataloader = dict(
    batch_size=1,
    num_workers=0,
    # persistent_workers=True,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTAv2Dataset',
        data_root='./data/split_ss_dota/',
        ann_file='subtrain/annfiles/',
        data_prefix=dict(img_path='subtrain/images/'),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
            dict(
                type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(
                type='ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='rbox')),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))

embedder_cfg = dict(type="resnet", embedder_dim=2048, batch_size=128)

anomaly_detector_cfg = dict(type="knn", k=[3, 5, 7, 11, 15, 21, 31, 51, 91, 99], use_cache=True, sample_ratio=0.5)

classifier_cfg = dict(type="vit", output_dir="train/OOD/vit-output/runs", per_device_train_batch_size=16,
                      per_device_eval_batch_size=16, evaluation_strategy="steps",
                      num_train_epochs=4, fp16=True, save_steps=0.1, eval_steps=0.1, logging_steps=0.2,
                      learning_rate=2e-4, save_total_limit=2, remove_unused_columns=False, push_to_hub=False,
                      report_to=['tensorboard'], load_best_model_at_end=True,
                      model_path='google/vit-base-patch16-224-in21k', retrain=False)

OOD_detector_cfg = dict(type="ODIN")
