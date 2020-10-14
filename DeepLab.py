import cv2
import onnxruntime as rt
import numpy as np

properties = {
    "xception65_ade20k_train": {'map': [127, 13], 'th': [0.45, 0.65]},
    "xception65_coco_voc_trainval": {'map': [12, 15], 'th': [0.35, 0.6]}
}


def to_square(im, desired_size, pad):
    old_size = im.shape[:2]
    desired_size -= (pad * 2)
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2 + pad, delta_h - (delta_h // 2) + pad
    left, right = delta_w // 2 + pad, delta_w - (delta_w // 2) + pad

    color = [0, 0, 0]
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), \
           cv2.copyMakeBorder(np.ones_like(im), top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), \
           (top, bottom, left, right, old_size)


def back_to_original_size(im, properties):
    im = im[properties[0]:-properties[1], properties[2]:-properties[3]]
    return cv2.resize(im, properties[4][::-1])


def process_pred(orig_pred, padding, th):
    pred = np.copy(orig_pred)
    pred[np.all(padding == 0, axis=2)] = 0
    pred[pred < th[0]] = 0
    pred[pred > th[1]] = 1
    pred_hesitant = np.zeros_like(pred)
    pred_hesitant[np.logical_and(pred > 0, pred < 1)] = 1
    return pred, pred_hesitant


SIZE = 512
PAD = 10


class DeepLab:
    def __init__(self, name):
        self.name = name
        self.sess = rt.InferenceSession(f'frozen_graphs/{self.name}.onnx')
        self.input_name = self.sess.get_inputs()
        self.label_name = self.sess.get_outputs()[0].name

    def eval(self, img):
        small_img, padding, resize_properties = to_square(img, SIZE, PAD)
        return self.sess.run([self.label_name], {self.input_name[0].name: np.expand_dims(small_img, 0)})[0], \
               small_img, padding, resize_properties

    def eval_specific_classes(self, img, classes):
        pred, small_img, padding, resize_properties = self.eval(img)
        ret = np.zeros_like(pred[:, :-1, :-1, 0])
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        for c in classes:
            ret += pred[:, :-1, :-1, c]
        ret = (ret - ret.min()) / (ret.max() - ret.min())
        return np.transpose(ret.astype(np.float), (1, 2, 0)), small_img, padding, resize_properties

    def eval_pepole_and_animal(self, img):
        return self.eval_specific_classes(img, properties[self.name]['map'])

    def get_props(self):
        return properties[self.name]