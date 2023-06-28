# 1. dataset settings
dataset_type = 'CocoDataset'
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
train_ann_root='../data/mimic-cxr-object/train_cxr_object.json'
val_ann_root='../data/mimic-cxr-object/val_cxr_object.json'
test_ann_root='../data/mimic-cxr-object/test_cxr_object.json'

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
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        ann_file=train_ann_root,
        data_prefix=dict(img=img_prefix)
        )
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        ann_file=val_ann_root,
        data_prefix=dict(img=img_prefix)
        )
    )

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        ann_file=test_ann_root,
        data_prefix=dict(img=img_prefix)
        )
    )

