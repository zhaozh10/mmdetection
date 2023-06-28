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
img_prefix='../data/mimic-cxr-resized/'
train_ann_root='../data/mimic-cxr-object/train_cxr_object_token.json'
val_ann_root='../data/mimic-cxr-object/val_cxr_object_token.json'
test_ann_root='../data/mimic-cxr-object/test_cxr_object_token.json'
IMAGE_INPUT_SIZE=512
# mean = 0.471
# std = 0.302

# A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
#             A.ColorJitter(hue=0.0),
#             A.GaussNoise(),
#             # randomly (by default prob=0.5) translate and rotate image
#             # mode and cval specify that black pixels are used to fill in newly created pixels
#             # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
#             A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
#             # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
#             A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
#             A.Normalize(mean=mean, std=std),

albu_train_transforms = [
    dict(
        type='LongestMaxSize',
        max_size=IMAGE_INPUT_SIZE,
        interpolation="cv2.INTER_AREA"
        ),
    dict(
        type='ColorJitter',
        hue=0.0,
        ),
    dict(
        type='GaussNoise',
        ),
    dict(
        type='Affine', 
        mode="cv2.BORDER_CONSTANT", 
        cval=0, 
        translate_percent=(-0.02, 0.02),
        rotate=(-2,2),
    ),
    dict(
        type="PadIfNeeded",
        min_height=IMAGE_INPUT_SIZE,
        min_width=IMAGE_INPUT_SIZE,
        border_mode="cv2.BORDER_CONSTANT"
    ),
    # dict(
    #     type="Normalize",
    #     mean=mean,
    #     std=std,
    # )
]
albu_val_transforms = [
    dict(
        type='LongestMaxSize',
        max_size=IMAGE_INPUT_SIZE,
        interpolation="cv2.INTER_AREA"
        ),
    dict(
        type="PadIfNeeded",
        min_height=IMAGE_INPUT_SIZE,
        min_width=IMAGE_INPUT_SIZE,
        border_mode="cv2.BORDER_CONSTANT"
    ),
    # dict(
    #     type="Normalize",
    #     mean=mean,
    #     std=std,
    # )
]
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='coco',
            label_fields=['gt_bboxes_labels']
        ),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),
    dict(type='PackDetInputs')
]
val_pipeline=[
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='Albu',
        transforms=albu_val_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='coco',
            label_fields=['gt_bboxes_labels']
        ),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),
    dict(type='PackDetInputs')
]

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
    batch_size=1,
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
    batch_size=1,
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