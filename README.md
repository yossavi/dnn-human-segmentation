# dnn-human-segmentation

Human segmantation and background remover using [DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab) and [Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/segmentation.html#semantic-segmentation) scene semantic segmantaion dnns.

For segmentation refinement the algorithm composes two postprocessing steps:
1. Assign pixels to either background or foreground using histograms similarity
2. Remove small connected components

*Currently, we support two models of deeplab using onnx that we created from the official freeze graph and two deeplab models using gluon-cv. It is optional to add more models with minor code changes.

# Installation
```pip install -r requirements.txt```

# Run
```
usage: main.py [-h] [-m MODEL_NAME] [-i IMAGES_DIR] [-o OUTPUT_DIR]
               [-b BG_BLEND] [-d PLOT_DEBUG] [-s OUTPUT_SIZE]

People Segmentation

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_NAME, --model_name MODEL_NAME
                        Name of segmentation model, one
                        of:(xception65_ade20k_train,
                        xception65_coco_voc_trainval, deeplab_resnet101_coco,
                        deeplab_resnet152_voc)
  -i IMAGES_DIR, --images_dir IMAGES_DIR
                        Path to images files
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to output folder
  -b BG_BLEND, --bg_blend BG_BLEND
                        OPTIONAL: bg image to blend. if None will plot
                        transpernet png
  -d PLOT_DEBUG, --plot_debug PLOT_DEBUG
                        Is to plot debug image
  -s OUTPUT_SIZE, --output_size OUTPUT_SIZE
                        Size of output image
  ``` 

# Results

<img src="https://github.com/yossavi/dnn-human-segmentation/blob/main/images/0pp.jfif" width="300"> <img src="https://github.com/yossavi/dnn-human-segmentation/blob/main/out_xception65_coco_voc_trainval/0pp.jfif_pastel.png.png" width="300">

<img src="https://github.com/yossavi/dnn-human-segmentation/blob/main/images/filipe-de-rodrigues-vetJrFdWesQ-unsplash.jpg" width="300"> <img src="https://github.com/yossavi/dnn-human-segmentation/blob/main/out_xception65_coco_voc_trainval/filipe-de-rodrigues-vetJrFdWesQ-unsplash.jpg_gradient.png.png" width="300">

<img src="https://github.com/yossavi/dnn-human-segmentation/blob/main/images/natasha-brazil-iFoMOlkWucI-unsplash.jpg" width="300"> <img src="https://github.com/yossavi/dnn-human-segmentation/blob/main/out_xception65_coco_voc_trainval/natasha-brazil-iFoMOlkWucI-unsplash.jpg_transparent.png.png" width="300">
