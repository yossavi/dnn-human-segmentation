import mxnet as mx
import numpy as np
from gluoncv.model_zoo import get_model

from DeepLab import to_square

properties = {
    'deeplab_resnet101_coco': {'map': [15], 'th': [0.5, 0.8]},
    'deeplab_resnet152_voc': {'map': [12, 15], 'th': [0.5, 0.8]}
}

SIZE = 512
PAD = 10


class Gluon:
    def __init__(self, name):
        self.name = name
        self.model = get_model(name, norm_layer=mx.gluon.nn.BatchNorm,
                               aux=False, base_size=520, crop_size=SIZE,
                               ctx=[mx.cpu(0)], pretrained=True)

    def eval(self, img):
        small_img, padding, resize_properties = to_square(img, SIZE, PAD)
        res = self.model(mx.nd.array(np.transpose(np.expand_dims(small_img / 128. - 1, 0), (0, 3, 1, 2))))
        res = np.squeeze(res[0].asnumpy())
        # res = np.exp(res) / np.exp(res).sum(axis=0)
        return res, small_img, padding, resize_properties

    def eval_specific_classes(self, img, classes):
        pred, small_img, padding, resize_properties = self.eval(img)
        ret = np.zeros_like(pred[0])
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        for c in classes:
            ret += pred[c]
        ret = (ret - ret.min()) / (ret.max() - ret.min())
        return ret, small_img, padding, resize_properties

    def eval_pepole_and_animal(self, img):
        return self.eval_specific_classes(img, properties[self.name]['map'])

    def get_props(self):
        return properties[self.name]
