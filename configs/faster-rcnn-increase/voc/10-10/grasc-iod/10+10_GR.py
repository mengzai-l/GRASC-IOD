_base_ = ['../10+10.py']

# ===================================================================
# 1. 基础设置
# ===================================================================
data_root = '/xxx/yyy/zzz/VOC2007'

# ===========================================================================
# 模型与训练策略
# ===========================================================================
model = dict(
    pseudo_label_setting=dict(
        is_use=True,
        alpha=1.,
    )
)

# ===========================================================================
# 2. Pipeline 设置
# ===========================================================================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# ===================================================================
# 3. 数据加载器 (Train Dataloader)
# ===================================================================
train_dataloader = dict(
    # 【核心！】加上这一行，彻底清除父配置里的 ignore_keys 和 ConcatDataset 设置
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
            ann_file='ImageSets/Main/final_merged_10_10.txt',
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
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0),
    type='OptimWrapper'
)

load_from = 'base/base10.pth'