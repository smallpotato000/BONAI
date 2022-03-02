# optimizer for 4 GPUs
optimizer = dict(type='SGD', lr=0.02/4, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.001,
    step=[16, 22])
total_epochs = 24
