# dnn-human-segmentation

Human segmantation and background remover using [DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab) and [Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/segmentation.html#semantic-segmentation) scene semantic segmantaion dnns.
For segmentation refinement the algorithm composes two postprocessing steps:
1. Assign pixels to either background or foreground using histograms similarity
2. Remove small connected components

# Installation
```pip install -r requirements.txt```

# Run
```python main.py``` 

# Results

