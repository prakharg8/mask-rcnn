from mmcv import Config
cfg2 = Config.fromfile('./configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py')

from mmdet.apis import set_random_seed

cfg2.dataset_type = 'COCODataset'

cfg2.total_epochs = 2
cfg2.runner['max_epochs'] = 2

cfg2.data.test.ann_file = '../drive/MyDrive/5/val/ann_coco.json'
cfg2.data.test.img_prefix = '../drive/MyDrive/5/input/'
cfg2.data.test.classes = ('panel', 'red_panel',)

cfg2.data.train.ann_file = '../drive/MyDrive/5/train/ann_coco.json'
cfg2.data.train.img_prefix = '../drive/MyDrive/5/input/'
cfg2.data.train.classes = ('panel', 'red_panel',)


cfg2.data.val.ann_file = '../drive/MyDrive/5/val/ann_coco.json'
cfg2.data.val.img_prefix = '../drive/MyDrive/5/input/'
cfg2.data.val.classes = ('panel', 'red_panel',)

# modify num classes of the model in box head and mask head
""" CHANGE THIS FOR SOLO AND SOLOV2 """
cfg2.model.roi_head.bbox_head.num_classes = 2
cfg2.model.roi_head.mask_head.num_classes = 2

# We can still use the pre-trained Mask RCNN model to obtain a higher performance
cfg2.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set up working dir to save files and logs.
cfg2.work_dir = './save_files_and_logs'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg2.optimizer.lr = 0.02 / 8
cfg2.lr_config.warmup = None
cfg2.log_config.interval = 10

# We can set the evaluation interval to reduce the evaluation times
cfg2.evaluation.interval = 12
# We can set the checkpoint saving interval to reduce the storage cost
cfg2.checkpoint_config.interval = 12

# Set seed thus the results are more reproducible
cfg2.seed = 0
set_random_seed(0, deterministic=False)
cfg2.gpu_ids = range(1)

# We can also use tensorboard to log the training process
cfg2.log_config.hooks = [
            dict(type='TextLoggerHook'),
                dict(type='TensorboardLoggerHook')]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg2.pretty_text}')

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


# Build dataset
datasets = [build_dataset(cfg2.data.train)]

# Build the detector
model2 = build_detector(cfg2.model)

# Add an attribute for visualization convenience
model2.CLASSES = datasets[0].CLASSES

# Create work_dir
cfg2.device='cuda'
mmcv.mkdir_or_exist(osp.abspath(cfg2.work_dir))
train_detector(model2, datasets, cfg2, distributed=False, validate=True)
