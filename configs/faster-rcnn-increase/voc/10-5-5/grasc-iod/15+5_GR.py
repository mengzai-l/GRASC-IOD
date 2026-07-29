_base_ = ['../15+5.py']

base_dir='work_dir/voc/10-5-5/grasc-iod/'
work_dir= base_dir + '15+5_GR'

ori_checkpoint = base_dir + '10+5_GR/epoch_6.pth'
ori_config='base/15.py'

data_root = '/xxx/yyy/zzz/VOC2007'

model = dict(
    pseudo_label_setting=dict(
        is_use=True,
        alpha=1.,
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='VOCDataset',
            data_root=data_root,
            ann_file='ImageSets/Main/final_merged_15_5.txt',
            metainfo=dict(
                classes=(
                    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
                )
            ),
            data_prefix=dict(sub_data_root='JPEGImages'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=None
        )
    )
)

train_cfg = dict(type='GPR', max_epochs=6, val_interval=1, lamda=0.6, is_use=True)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=6, by_epoch=True, milestones=[3, 5], gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0),
    type='OptimWrapper'
)

ori_checkpoint = base_dir + '10+5_GR/epoch_6.pth'
