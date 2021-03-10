# Using Model for Inference
## Load Model Weights from trained weights
## Predictor Setup
## Predict and Visualize the result

from detectron2.utils.visualizer import ColorMode
import glob
import cv2
import matplotlib.pyplot as plt

# inference
def using_model(metadata, threshold):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth') # 나중에 용주님 토치서브로 넘길것
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   # set the testing threshold for this model
    cfg.DATASETS.TEST = (metadata.name, )
    predictor = DefaultPredictor(cfg)

    images = [cv2.imread(file) for file in glob.glob("/gdrive/MyDrive/img/*.png")] # 이미지 경로, 확장자 수정

    dataset_dicts = DatasetCatalog.get(metadata.name)
    for i in range(len(images)):
        im = images[i]
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()