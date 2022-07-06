import mmcv
import os.path as osp
from tqdm import tqdm
def convert_data_to_coco(ann_files, img_dirs, out_file_train, out_file_val, out_img_train, out_img_val):
    
    dic = {
        'panel': 0,
        'red_panel': 1
    }
    #"""
    #test
    images = []
    annotations = []
    obj_count = 0
    for (ann_file, imgs) in zip(ann_files, img_dirs):
        data_infos = mmcv.load(ann_file)
        total = len(data_infos['assets'].values())
        
        for idx, v in tqdm(zip(range(int(total * 0.8)), data_infos['assets'].values())): # range(total * 0.8, total) for val (used zip to change the number of images used for train and val)
            filename = v['asset']['name']
            #filename = filename[:-3] + 'png'
            img_path = osp.join(imgs, filename)
            # if idx == 0: print(img_path)
            if not osp.isfile(img_path): continue
            img = mmcv.image.imread(img_path)
            new_img_path = osp.join(out_img_train, filename)
            mmcv.image.imwrite(img, new_img_path)
            height = 1080
            width = 1920

            images.append(dict(
                id=idx,
                file_name=filename,
                height=height,
                width=width))

            bboxes = []
            labels = []
            masks = []
            for obj in v['regions']:
                if obj['tags'][0] == 'dicey': continue
                obj_cat = dic[obj['tags'][0]]
                px = [a['x'] for a in obj['points']]
                py = [a['y'] for a in obj['points']]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                x_min, y_min, x_max, y_max = (
                    min(px), min(py), max(px), max(py))


                data_anno = dict(
                    image_id=idx,
                    id=obj_count,
                    category_id=obj_cat,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=(x_max - x_min) * (y_max - y_min),
                    segmentation=[poly],
                    iscrowd=0)
                annotations.append(data_anno)
                obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':0, 'name': 'panel'}, {'id':1, 'name': 'red_panel'}])
    mmcv.dump(coco_format_json, out_file_train)
    """

    # val
    annotations = []
    images = []
    obj_count = 0
    for (ann_file, imgs) in zip(ann_files, img_dirs):
        data_infos = mmcv.load(ann_file)
        total = len(data_infos['assets'].values())
        #print(total)
        for idx, v in tqdm(zip(range(int(total * 0.8), total), data_infos['assets'].values())):
            # print(v['asset']['path'])
            filename = v['asset']['name']
            #filename = filename[:-3] + 'png'
            img_path = osp.join(imgs, filename)
            if not osp.isfile(img_path): continue
            #if idx == 0: print(img_path)
            img = mmcv.image.imread(img_path)
            new_img_path = osp.join(out_img_val, filename)
            mmcv.image.imwrite(img, new_img_path)
            height = 1080
            width = 1920

            images.append(dict(
                id=idx,
                file_name=filename,
                height=height,
                width=width))

            bboxes = []
            labels = []
            masks = []

            for obj in v['regions']:
                if obj['tags'][0] == 'dicey': continue
                obj_cat = dic[obj['tags'][0]]
                px = [a['x'] for a in obj['points']]
                py = [a['y'] for a in obj['points']]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                x_min, y_min, x_max, y_max = (
                    min(px), min(py), max(px), max(py))


                data_anno = dict(
                    image_id=idx,
                    id=obj_count,
                    category_id=obj_cat,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=(x_max - x_min) * (y_max - y_min),
                    segmentation=[poly],
                    iscrowd=0)
                annotations.append(data_anno)
                obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id':0, 'name': 'panel'}, {'id':1, 'name': 'red_panel'}])
    mmcv.dump(coco_format_json, out_file_val)
    """

# 1 10 11 2 3 4 5 6 9
ann_files = ['../../Data/1/export.json', '../../Data/10/export.json', '../../Data/11/export.json', '../../Data/2/export.json', '../../Data/3/export.json', '../../Data/4/export.json', '../../Data/5/export.json', '../../Data/6/export.json', '../../Data/9/export.json'] # all annotation jsons
imgs = ['../../Data/1/', '../../Data/10/', '../../Data/11/', '../../Data/2/', '../../Data/3/', '../../Data/4/', '../../Data/5/', '../../Data/6/', '../../Data/9/'] # paths to all images
# (all annotations list, all images list, train annotations, val annotations) 
convert_data_to_coco(ann_files, imgs, './train/ann_coco.json', './val/ann_coco.json', './train/images', './val/images/')
    
