# from mmdet.datasets import MIMIC_Object_Dataset

# img_prefix='../data/mimic-cxr-resized/'
# train_ann_root='../GroundingDINO/train_cxr_object_token.json'
# val_ann_root='../data/mimic-cxr-object/val_cxr_object.json'
# test_ann_root='../data/mimic-cxr-object/test_cxr_object.json'

# dataset=MIMIC_Object_Dataset(ann_file=train_ann_root,data_prefix={'img':img_prefix})
# ret=dataset.__getitem__(idx=0)
# print("hold")


from mmdet.datasets.transforms.transforms import Albu

IMAGE_INPUT_SIZE=512
mean = 0.471
std = 0.302

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
    dict(
        type="Normalize",
        mean=mean,
        std=std,
    )
]


ATrans=Albu(transforms=albu_train_transforms,
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
        skip_img_without_anno=True)

print("hold")

