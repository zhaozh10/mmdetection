from mmdet.datasets import MIMIC_Object_Dataset

img_prefix='../data/mimic-cxr-resized/'
train_ann_root='../GroundingDINO/train_cxr_object_token.json'
val_ann_root='../data/mimic-cxr-object/val_cxr_object.json'
test_ann_root='../data/mimic-cxr-object/test_cxr_object.json'

dataset=MIMIC_Object_Dataset(ann_file=train_ann_root,data_prefix={'img':img_prefix})
ret=dataset.__getitem__(idx=0)
print("hold")



