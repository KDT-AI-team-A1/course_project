# Using Model for Inference
## Load Model Weights from trained weights
## Predictor Setup
## Predict and Visualize the result
from detectron2.utils.visualizer import ColorMode
import glob
import cv2
import matplotlib.pyplot as plt
import time
times = []
# inference
def using_model(metadata, threshold):

  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   
  cfg.DATASETS.TEST = (metadata.name, )
  predictor = DefaultPredictor(cfg)

  images = [cv2.imread(file) for file in glob.glob("/gdrive/MyDrive/img/*.png")]

  dataset_dicts = DatasetCatalog.get(metadata.name)
  for i in range(len(images)):
    start_time = time.time()
    im = images[i]
    outputs = predictor(im)
    delta = time.time() - start_time
    times.append(delta)
    v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()
  mean_delta = np.array(times).mean()
  fps = 1 / mean_delta
  print("평균 추론 sec:{:.2f},fps:{:.2f}".format(mean_delta, fps)) # 평균 추론 속도
  print(outputs['instances'].pred_classes)
  print(outputs['instances'].pred_boxes)