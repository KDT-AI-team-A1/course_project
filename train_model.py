# Train Model
## Setup Configuration for Detectron2 Model
## Trainer Setup & Train

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
# import PIL

"""
함수 사용할 시
cfg = train_model(metadata = mask_metadata,
            model_dir= "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
            is_mask= False,
            num_workers= 4,
            batch_size= 256,
            max_iter= 200,
            num_classes= 3)
"""

def train_model(metadata, model_dir, is_mask, num_workers, batch_size, max_iter, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_dir))
    cfg.DATASETS.TRAIN = (metadata.name,)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_dir)
    cfg.MODEL.MASK_ON = is_mask
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    return cfg