# Using Model for Inference
## Load Model Weights from trained weights
## Predictor Setup
## Predict and Visualize the result

from detectron2.utils.visualizer import ColorMode
import random

# inference
def using_model(metadata, threshold):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   # set the testing threshold for this model
    cfg.DATASETS.TEST = (metadata.name, )
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get(metadata.name)
    for d in random.sample(dataset_dicts, 5):    
        im = cv2.imread(d['file_name'])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8)
        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        plt.figure(figsize = (14, 10))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        plt.show()