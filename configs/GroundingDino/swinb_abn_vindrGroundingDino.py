# Inherit and overwrite part of the config based on this config
_base_ = 'configs/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py'

data_root = '/home/woody/iwi5/iwi5204h/mmdetection/data/coco/' # dataset root
im_root = '/home/woody/iwi5/iwi5204h/mmdetection/data/VinDir/Train/'
test_root = '/home/woody/iwi5/iwi5204h/mmdetection/data/VinDir/Test/Converted_to_jpeg/'

train_batch_size_per_gpu = 2 #4 gives oom
train_num_workers = 4

max_epochs = 37
stage2_num_epochs = 1
base_lr = 0.00008

metainfo = {
    'classes': (
        'infiltration', 'lung opacity', 'lung cyst', 'rib fracture', 
        'pleural effusion', 'lung cavity', 'calcification', 
        'mediastinal shift', 'atelectasis', 'cardiomegaly', 'emphysema', 
        'eventration', 'pulmonary fibrosis', 'consolidation', 'no finding', 
        'pneumothorax', 'other lesion', 'pleural thickening', 
        'aortic enlargement', 'enlarged pa', 'clavicle fracture', 
        'ild', 'nodule/mass', 'edema'
    ),
    'palette': [
        (220, 120, 60), (0, 11, 255), (0, 0, 142), (0, 0, 230), 
        (106, 0, 228), (100, 170, 30), (139, 0, 139), (255, 0, 0), 
        (255, 20, 147), (240, 128, 128), (255, 192, 203), (250, 128, 114), 
        (255, 105, 180), (255, 99, 71), (255, 69, 0), (240, 128, 128), 
        (240, 128, 128), (240, 128, 128), (205, 92, 92), (255, 0, 255), 
        (128, 0, 0), (255, 50, 0), (128, 0, 128), (139, 69, 19)
    ]
}

model = dict(bbox_head=dict(num_classes=24))
model = dict(
    bbox_head=dict(
        num_classes=24,
        loss_feat=dict(type='FeatureReconstructLoss', loss_weight=0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)))


# over-write `train_pipeline` for new added `AutoAugment` training setting
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', scale=1024, keep_ratio=True),
    dict(type='AutoContrast', prob=0.5),
    dict(type='Sharpness', prob=0.5),
    #dict(type='Equalize', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    #scales=[(400, 4200), (500, 4200), (600, 4200)],
                    scales=[(800, 4200)],
                    keep_ratio=True),
                
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),

    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
#    dict(type='Normalize', **img_norm_cfg),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]

data = dict(
    samples_per_gpu= train_batch_size_per_gpu,
    workers_per_gpu= train_num_workers,
    # over-write `pipeline` with new training pipeline setting
    train=dict(dataset=dict(pipeline=train_pipeline))
    )

    

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        #data_root='/',
        pipeline=train_pipeline,
        metainfo=metainfo,
        data_prefix=dict(img=im_root),
        ann_file=data_root+'train_abn_vindr_16k_orig.json'))

val_dataloader = dict(
    dataset=dict(
        #data_root='/',
        pipeline = test_pipeline, 
        metainfo=metainfo,
        data_prefix=dict(img=test_root),
        ann_file=data_root+'test_abn_vindr_16k_orig.json'))

val_evaluator = dict(ann_file=data_root + 'test_abn_vindr_16k_orig.json', classwise=True)
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=base_lr),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.1),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))

default_hooks = dict(
    checkpoint=dict(
        interval=1,
#        max_keep_ckpts=3,  # only keep latest 2 checkpoints
#        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=50))

auto_scale_lr = dict(base_batch_size=4)
model = dict(test_cfg=dict(max_per_img=100))

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
