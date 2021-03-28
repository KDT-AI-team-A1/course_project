import pickle
from fvcore.common.file_io import PathManager
import itertools
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN
import logging
import os
import os.path as osp
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, DefaultPredictor
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.modeling import GeneralizedRCNNWithTTA, build_model
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from tqdm import tqdm

import easydict
from detectron2 import model_zoo

logger = logging.getLogger(__name__)
_C.MY_CUSTOM = CN()

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def get_detection_dataset_dicts(dataset_names, filter_empty=True, min_keypoints=0, proposal_files=None):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.
    Args:
        dataset_names (list[str]): a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.
    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if has_instances:
        try:
            check_metadata_consistency("thing_classes", dataset_names)
        except AttributeError:  # class names are not available for this dataset
            pass
    return dataset_dicts


class AdetCheckpointer(DetectionCheckpointer):
    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        if "lpf" in filename:
            loaded["matching_heuristics"] = True
        return loaded


def build_detection_val_loader(cfg, dataset_name: str, mapper=None):
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    dataset = DatasetFromList(dataset_dicts)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def parse_args():
    args = easydict.EasyDict({
        "gpu": 0,
        "resume": True,
        "output": "/content/output/",
        "max_iter": 300,
        "num_machines": 1,
        "eval_only": False,
        "learning_rate": 0.0005,
        "ims_per_batch": 2,
        "batch_size_per_image": 512,
        "num_classes": 2,
        "eval_period": 100
    })
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    return args



def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)



def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_val(cfg, model, val_dataloader):
    with torch.no_grad():
        losses = []
        tmp = None
        for idx, inputs in tqdm(enumerate(val_dataloader)):
            outputs = model(inputs)
            loss_dict_reduced = {
                k: v.item() for k, v in comm.reduce_dict(outputs).items()
            }
            tmp = loss_dict_reduced
            reduced_loss = sum(loss for loss in loss_dict_reduced.values())
            losses.append(reduced_loss)
    val_loss = np.mean(losses)
    return val_loss


def do_train(cfg, model, resume=True, val_set='mask_val'):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    print_every = 50

    tensorboard_dir = osp.join(cfg.OUTPUT_DIR, 'tensorboard')
    checkpoint_dir = osp.join(cfg.OUTPUT_DIR, 'checkpoints')

    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpointer = AdetCheckpointer(
        model, checkpoint_dir, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            TensorboardXWriter(tensorboard_dir),
        ]
        if comm.is_main_process()
        else []
    )
    data_loader = build_detection_train_loader(cfg)
    val_dataloader = build_detection_val_loader(cfg, val_set)

    logger.info("Starting training from iteration {}".format(start_iter))

    # [PHAT]: Create a log file
    log_file = open(cfg.MY_CUSTOM.LOG_FILE, 'w')

    best_loss = 100
    count_not_improve = 0
    # train_size = 6690
    # epoch_size = int(train_size/cfg.SOLVER.IMS_PER_BATCH)
    epoch_size = 100
    n_early_epoch = 2

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())

            assert torch.isfinite(losses).all(), loss_dict

            # Update loss dict
            loss_dict_reduced = {
                k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            # Early stopping
            if (iteration > start_iter) and ((iteration - start_iter) % epoch_size == 0):

                val_loss = do_val(cfg, model, val_dataloader)

                if val_loss >= best_loss:
                    count_not_improve += 1
                    # stop if models doesn't improve after <n_early_epoch> epoch
                    # if count_not_improve == epoch_size*n_early_epoch:
                    if count_not_improve == n_early_epoch:
                        break
                else:
                    count_not_improve = 0
                    best_loss = val_loss
                    periodic_checkpointer.save("best_model_early")

                log_file.write(f"Epoch {(iteration - start_iter) // epoch_size}, val_loss: {val_loss}\n")
                comm.synchronize()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]["lr"]
            storage.put_scalar("lr", lr, smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and ((iteration - start_iter) % print_every == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
                # Write my log
                log_file.write(f"[iter {iteration}, best_loss: {best_loss}] total_loss: {losses}, lr: {lr}\n")
            periodic_checkpointer.step(iteration)
    log_file.close()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.MASK_ON = False
    cfg.DATASETS.TRAIN = ('mask_train',)
    cfg.DATASETS.TEST = ('mask_val',)
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MY_CUSTOM.LOG_FILE = os.path.join(cfg.OUTPUT_DIR, 'my_log.txt')
    cfg.OUTPUT_DIR = args.output
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.freeze()
    return cfg


def main():
    args = parse_args()
    ltd = Load_Train_Data(DIR_INPUT + 'train_2.csv', DIR_TRAIN)
    clear_dataset_catalog()
    register_dataset_catalog(ltd, phase=['mask_train'], classes=['with', 'No'])
    vtd = Load_Train_Data(DIR_INPUT + 'val_2.csv', DIR_TRAIN)
    register_dataset_catalog(vtd, phase=['mask_val'], classes=['with', 'No'])

    cfg = setup(args)
    model = build_model(cfg)

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, val_set='mask_val')
    return do_test(cfg, model)


if __name__ == "__main__":
    main()