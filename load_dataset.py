import pandas as pd
import numpy as np
from detectron2.structures import BoxMode
from tqdm import tqdm
import os
import cv2

annotation_file = 'train_all.csv'
DIR_IMAGE = '/images'


class Load_Train_Data:
    def __init__(self, annotation_file, IMAGE_DIR):
        self.df = pd.read_csv(annotation_file)
        self.DIR = IMAGE_DIR

    def generate_box(self, series):
        xmin = int(series['x1'])
        ymin = int(series['y1'])
        xmax = int(series['x2'])
        ymax = int(series['y2'])

        return [xmin, ymin, xmax, ymax]

    def generate_label(self, series):
        if series['classname'] == 'face_with_mask':
            return 0
        elif series['classname'] == 'face_with_mask_incorrect':
            return 1
        elif series['classname'] == 'face_no_mask':
            return 2

    def generate_target(self, filename):
        target_df = self.df[self.df['name'] == filename]
        return_objs = []
        for i in range(len(target_df)):
            target = {'bbox': self.generate_box(target_df.iloc[i]),
                      'bbox_mode': BoxMode.XYXY_ABS,
                      'category_id': self.generate_label(target_df.iloc[i])}
            return_objs.append(target)
        return return_objs

    def get_mask_dicts(self, df):
        dataset_dicts = []
        image_name_list = df['name'].unique()
        for idx, image_name in enumerate(tqdm(image_name_list)):
            filename = os.path.join(DIR_IMAGE, image_name)
            objs = self.generate_target(image_name)
            target_image = df[df['name'] == image_name]

            record = {"file_name": filename,
                      "height": target_image['width'].iloc[0],
                      "width": target_image['height'].iloc[0],
                      'image_id': idx,
                      'annotations': objs}

            dataset_dicts.append(record)
        return dataset_dicts
