import pandas as pd
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from tqdm import tqdm
import os
import cv2


def register_dataset_catalog(loader, phase, classes):
    """
    Register Dataset Catalog for training/validation(test)
    :param loader: Load_Train_Data class
    :param phase: 'train' or 'val' or user define phase
    :param classes: list of classes used in dataset
    :return: Nothing
    """
    for p in phase:
        DatasetCatalog.register(p, lambda p=p: loader.get_mask_dicts())
        MetadataCatalog.get(p).set(thing_classes=classes)


def clear_dataset_catalog():
    """
    Clear dataset/metadata catalog
    :return: Nothing
    """
    DatasetCatalog.clear()
    MetadataCatalog.clear()


class Load_Train_Data:
    """
    Data loading Class
    """
    def __init__(self, annotation_file, IMAGE_DIR):
        """
        Initialize class
        :param annotation_file:  file route of annotation information file
        :param IMAGE_DIR: Dataset Image file route
        """
        self.df = pd.read_csv(annotation_file) # load annotation file as pd.DataFrame
        self.DIR = IMAGE_DIR

    def generate_box(self, series):
        """
        generate box info from pd.Series
        :param series: pd.Series from self.df
        :return: list of bbox info in formax (XYXY_ABS)
        """
        xmin = int(series['x1'])
        ymin = int(series['y1'])
        xmax = int(series['x2'])
        ymax = int(series['y2'])

        return [xmin, ymin, xmax, ymax]

    def generate_label(self, series):
        """
        generate class label from pd.Series
        :param series: pd.Series from self.df
        :return: value of class
        """
        if series['classname'] == 'face_with_mask':
            return 0
        elif series['classname'] == 'face_with_mask_incorrect':
            return 1
        elif series['classname'] == 'face_no_mask':
            return 2

    def generate_target(self, filename):
        """
        generate targer dict for one file
        :param filename: image filename
        :return: bboxes for image file
        """
        target_df = self.df[self.df['name'] == filename]
        return_objs = []
        for i in range(len(target_df)):
            target = {'bbox': self.generate_box(target_df.iloc[i]), # generate bbox
                      'bbox_mode': BoxMode.XYXY_ABS, # boxmode == XYXY_ABS
                      'category_id': self.generate_label(target_df.iloc[i])} # generagte class label
            return_objs.append(target)
        return return_objs

    def get_mask_dicts(self):
        """
        make annotation dictionary for dataset
        :return: dataset annotation dictionary
        """
        dataset_dicts = []
        image_name_list = self.df['name'].unique()
        for idx, image_name in enumerate(tqdm(image_name_list)):
            filename = os.path.join(self.DIR, image_name)
            objs = self.generate_target(image_name)
            target_image = self.df[self.df['name'] == image_name]

            record = {"file_name": filename,
                      "height": target_image['height'].iloc[0],
                      "width": target_image['width'].iloc[0],
                      'image_id': idx,
                      'annotations': objs}

            dataset_dicts.append(record)
        return dataset_dicts

