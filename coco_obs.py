from mmdet.datasets.api_wrappers import COCO
import os.path as osp
import copy



def parse_data_info(raw_data_info: dict,cat_ids, cat2label,):
    """Parse raw annotation to target format.

    Args:
        raw_data_info (dict): Raw data information load from ``ann_file``

    Returns:
        Union[dict, List[dict]]: Parsed annotation.
    """
    img_info = raw_data_info['raw_img_info']
    ann_info = raw_data_info['raw_ann_info']

    data_info = {}

    # TODO: need to change data_prefix['img'] to data_prefix['img_path']
    img_path = osp.join(data_prefix['img'], img_info['file_name'])
    if data_prefix.get('seg', None):
        seg_map_path = osp.join(
            data_prefix['seg'],
            img_info['file_name'].rsplit('.', 1)[0])
    else:
        seg_map_path = None
    data_info['img_path'] = img_path
    data_info['img_id'] = img_info['img_id']
    data_info['seg_map_path'] = seg_map_path
    data_info['height'] = img_info['height']
    data_info['width'] = img_info['width']
    if img_info.get('report',None):
        data_info['report']=img_info['report']
    else:
        data_info['report']=None

    instances = []
    for i, ann in enumerate(ann_info):
        instance = {}

        if ann.get('ignore', False):
            continue
        x1, y1, w, h = ann['bbox']
        inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
        inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
        if inter_w * inter_h == 0:
            continue
        if ann['category_id'] not in cat_ids:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]

        if ann.get('iscrowd', False):
            instance['ignore_flag'] = 1
        else:
            instance['ignore_flag'] = 0
        instance['bbox'] = bbox
        instance['bbox_label'] = cat2label[ann['category_id']]
        instance['caption']=ann['caption']
        instance['input_ids']=ann['input_ids']

        if ann.get('segmentation', None):
            instance['mask'] = ann['segmentation']

        instances.append(instance)
    data_info['instances'] = instances
    return data_info






METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }

coco=COCO('../GroundingDINO/val_cxr_object_token.json')
METAINFO['classes']=classes = (
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
cat_ids = coco.get_cat_ids(cat_names=METAINFO['classes'])
cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
cat_img_map = copy.deepcopy(coco.cat_img_map)
img_ids = coco.get_img_ids()
data_prefix={}
data_prefix['img']='../data/mimic-cxr-resized/'





data_list = []
total_ann_ids = []
for img_id in img_ids:
    raw_img_info = coco.load_imgs([img_id])[0]
    raw_img_info['img_id'] = img_id

    ann_ids = coco.get_ann_ids(img_ids=[img_id])
    raw_ann_info = coco.load_anns(ann_ids)
    total_ann_ids.extend(ann_ids)

    parsed_data_info = parse_data_info({
        'raw_ann_info':
        raw_ann_info,
        'raw_img_info':
        raw_img_info
    },cat_ids=cat_ids,cat2label=cat2label)
    data_list.append(parsed_data_info)
print("hold")
