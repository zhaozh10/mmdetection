# 1. dataset settings
dataset_type = 'MIMIC_Object_Dataset'
classes = (
    "right lung",
    "right upper lung zone",
    "right mid lung zone",
    "right lower lung zone",
    "right hilar structures",
    "right apical zone",
    "right costophrenic angle",
    "right hemidiaphragm",
    "left lung",
    "left upper lung zone",
    "left mid lung zone",
    "left lower lung zone",
    "left hilar structures",
    "left apical zone",
    "left costophrenic angle",
    "left hemidiaphragm",
    "trachea",
    "spine",
    "right clavicle",
    "left clavicle",
    "aortic arch",
    "mediastinum",
    "upper mediastinum",
    "svc",
    "cardiac silhouette",
    "cavoatrial junction",
    "right atrium",
    "carina",
    "abdomen"
)
# img_prefix='../data/mimic-cxr-resized/'
# train_ann_root='../data/mimic-cxr-object/train_cxr_object_token.json'
# val_ann_root='../data/mimic-cxr-object/val_cxr_object_token.json'
# test_ann_root='../data/mimic-cxr-object/test_cxr_object_token.json'

img_prefix='/public_bme/data/physionet.org/files/mimic-cxr-jpg/2.0.0/'
train_ann_root='/public_bme/data/physionet.org/files/mimic-cxr-object/train_cxr_object_token.json'
val_ann_root='/public_bme/data/physionet.org/files/mimic-cxr-object/val_cxr_object_token.json'
test_ann_root='/public_bme/data/physionet.org/files/mimic-cxr-object/test_cxr_object_token.json'
IMAGE_INPUT_SIZE=512
backend_args = None
# mean = 0.471
# std = 0.302

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor','flip', 'flip_direction')
        )
]
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor')
        )
]


# albu_train_transforms = [
#     dict(
#         type='LongestMaxSize',
#         max_size=IMAGE_INPUT_SIZE,
#         interpolation=3
#         ),
#     dict(
#         type='ColorJitter',
#         hue=0.0,
#         ),
#     dict(
#         type='GaussNoise',
#         ),
#     dict(
#         type='Affine', 
#         mode=0, 
#         cval=0, 
#         translate_percent=(-0.02, 0.02),
#         rotate=(-2,2),
#     ),
#     dict(
#         type="PadIfNeeded",
#         min_height=IMAGE_INPUT_SIZE,
#         min_width=IMAGE_INPUT_SIZE,
#         border_mode=0
#     ),
# ]
# albu_val_transforms = [
#     dict(
#         type='LongestMaxSize',
#         max_size=IMAGE_INPUT_SIZE,
#         interpolation=3
#         ),
#     dict(
#         type="PadIfNeeded",
#         min_height=IMAGE_INPUT_SIZE,
#         min_width=IMAGE_INPUT_SIZE,
#         border_mode=0
#     ),
# ]

# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
#     dict(
#         type='Albu',
#         transforms=albu_train_transforms,
#         bbox_params=dict(
#             type='BboxParams',
#             format='pascal_voc',
#             label_fields=['gt_bboxes_labels']
#         ),
#         keymap={
#             'img': 'image',
#             'gt_bboxes': 'bboxes'
#         },),
#     dict(type='PackDetInputs',meta_keys=('img_id', 'img_path', 'img_shape',
#                    'report','gt_bboxes_labels',"gt_bboxes"))
# ]
# val_pipeline=[
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
#     dict(
#         type='Albu',
#         transforms=albu_val_transforms,
#         bbox_params=dict(
#             type='BboxParams',
#             format='pascal_voc',
#             label_fields=['gt_bboxes_labels']
#         ),
#         keymap={
#             'img': 'image',
#             'gt_bboxes': 'bboxes'
#         },
#         ),
#     dict(type='PackDetInputs',meta_keys=('img_id', 'img_path', 'img_shape',
#                    'report','gt_bboxes_labels',"gt_bboxes"))
# ]

"""Args:
    ann_file (str, optional): Annotation file path. Defaults to ''.
    metainfo (dict, optional): Meta information for dataset, such as class
        information. Defaults to None.
    data_root (str, optional): The root directory for ``data_prefix`` and
        ``ann_file``. Defaults to ''.
    data_prefix (dict): Prefix for training data. Defaults to
        dict(img_path='').
    filter_cfg (dict, optional): Config for filter data. Defaults to None.
    indices (int or Sequence[int], optional): Support using first few
        data in annotation file to facilitate training/testing on a smaller
    serialize_data (bool, optional): Whether to hold memory using
        serialized objects, when enabled, data loader workers can use
        shared RAM from master process instead of making a copy. Defaults
        to True.
    pipeline (list, optional): Processing pipeline. Defaults to [].
    test_mode (bool, optional): ``test_mode=True`` means in test phase.
        Defaults to False.
    lazy_init (bool, optional): Whether to load annotation during
        instantiation. In some cases, such as visualization, only the meta
        information of the dataset is needed, which is not necessary to
        load annotation file. ``Basedataset`` can skip load annotations to
        save time by set ``lazy_init=True``. Defaults to False.
    max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
        None img. The maximum extra number of cycles to get a valid
        image. Defaults to 1000.
"""
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        ann_file=train_ann_root,
        data_prefix=dict(img=img_prefix),
        pipeline=train_pipeline,
        )
    )

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        ann_file=val_ann_root,
        data_prefix=dict(img=img_prefix),
        pipeline=val_pipeline
        )
    )

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        ann_file=test_ann_root,
        data_prefix=dict(img=img_prefix),
        pipeline=val_pipeline
        )
    )

val_evaluator = dict(
    type='CocoMetric',
    ann_file=val_ann_root,
    metric=['bbox'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator