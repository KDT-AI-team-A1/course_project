# Train Model
## Setup Configuration for Detectron2 Model
## Trainer Setup & Train

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

def train_model(metadata, model_dir, num_workers, batch_size, max_iter, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_dir))
    cfg.DATASETS.TRAIN = (metadata.name,)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_dir)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()