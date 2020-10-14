# python -m tf2onnx.convert --input frozen_graphs/deeplabv3_pascal_trainval/frozen_inference_graph.pb --output frozen_graphs/xception65_coco_voc_trainval.onnx --outputs ResizeBilinear_3:0 --inputs ImageTensor:0 --opset 12
# python -m tf2onnx.convert --input frozen_graphs/deeplabv3_xception_ade20k_train/frozen_inference_graph.pb --output frozen_graphs/xception65_ade20k_train.onnx --outputs ReimsizeBilinear_3:0 --inputs ImageTensor:0 --opset 12
# python -m tf2onnx.convert --input frozen_graphs/trainval_fine/frozen_inference_graph.pb --output frozen_graphs/xception71_dpc_cityscapes_trainval.onnx --outputs Mean:0 --inputs ImageTensor:0 --opset 12
# python -m tf2onnx.convert --checkpoint frozen_graphs/efficientnet-l2-nasfpn-ssl/model.ckpt.meta --output frozen_graphs/efficientnet-l2-nasfpn-ssl.onnx --outputs Mean:0 --inputs ImageTensor:0 --opset 12

import argparse
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from DeepLab import DeepLab, back_to_original_size, process_pred
from Gluon import Gluon


def laplacian_pyramid_blending(A, B, m, num_levels=6):
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    lpA = [gpA[num_levels - 1]]
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]
    for i in range(num_levels - 1, 0, -1):
        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])  # also reverse the masks

    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    ls_ = LS[0]
    for i in range(1, num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = ls_ + LS[i]

    return ls_


def get_charactistic_colors(img):
    kmeans = KMeans(n_clusters=10).fit(img.reshape((-1, 3)))
    centers = kmeans.cluster_centers_
    centers_count = np.bincount(kmeans.labels_)
    centers_probs = centers_count / centers_count.sum()
    return centers, centers_probs


def get_similarity(pixel, centers, centers_probs):
    return ((np.sqrt(3) - np.linalg.norm(np.expand_dims(pixel, 2) - centers, axis=3)) * centers_probs).sum(axis=2) / \
           centers_probs.shape[0]


def get_mask(pred, bg_fg_sim_diff, csf):
    mask = np.copy(pred)
    mask += bg_fg_sim_diff / bg_fg_sim_diff.max() * csf
    return np.clip(mask, 0, 1)


def remove_small_connected_componnets(mask):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats((mask * 255.).astype(np.uint8),
                                                                               connectivity=8)
    sizes = stats[1:, -1]
    for i in range(nb_components - 1):
        if sizes[i] < sizes.max() * 0.8:
            mask[output == i + 1] = 0

    return mask


def get_bg(path, size):
    bg = None
    if path is not None:
        bg = cv2.imread(path)
        if bg is not None:
            bg = cv2.resize(bg, size)
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            bg = bg / 255.
    return bg


def blend_img_bg(img, bg, mask, num_levels=1):
    if bg is None:
        return np.concatenate((img, mask), axis=2)
    if num_levels == 1:
        return np.concatenate((img * mask + bg * (1 - mask), np.ones_like(mask)), axis=2)
    else:
        return laplacian_pyramid_blending(img, bg, np.transpose(np.squeeze(np.array([mask, mask, mask])), (1, 2, 0)),
                                          num_levels=num_levels)


def get_grad(img):
    g = cv2.cvtColor((cv2.blur(img, (3, 3), 0) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    grads = np.sqrt(np.power(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3), 2) + np.power(
        cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3), 2))
    return (grads - grads.min()) / (grads.max() - grads.min())


def nearest_class(i, j, stop, mask, grads):
    if stop == 1:
        return 1, -1
    if grads[i, j] > 0.2:
        return 1, -1
    if mask[i, j] == 0 or mask[i, j] == 1:
        return 2, mask[i, j]
    return 0, -1


def clear_mask_by_grads(mask, grads):
    ret = np.copy(mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if 1 > mask[i, j] > 0:
                stop = np.zeros(4)
                cls = np.zeros(4)
                for t in range(30):
                    stop[0], cls[0] = nearest_class(i + t, j, stop[0], mask, grads)
                    stop[1], cls[1] = nearest_class(i, j + t, stop[1], mask, grads)
                    stop[2], cls[2] = nearest_class(i - t, j, stop[2], mask, grads)
                    stop[3], cls[3] = nearest_class(i, j - t, stop[3], mask, grads)
                    if np.any(stop == 2):
                        ret[i, j] = cls.max()
                        break
    return ret


def plot_debug(img, bg_centers, fg_centers, pred, pred_th, bg_fg_sim_diff, mask, blend, grads):
    fig, axs = plt.subplots(2, 4)
    for a in axs.flatten():
        a.axis('off')

    axs[0, 0].imshow(img)
    axs[0, 1].imshow(pred, cmap='Greys_r')
    axs[0, 2].imshow(pred_th, cmap='Greys_r')
    axs[0, 3].imshow(np.array([bg_centers, fg_centers]))

    axs[1, 0].imshow(bg_fg_sim_diff, cmap='Greys_r')
    axs[1, 1].imshow(mask, cmap='Greys_r')
    axs[1, 2].imshow(blend)
    axs[1, 3].imshow(grads, cmap='Greys_r')
    plt.savefig(os.path.join(args.output_dir, "debug", filename + ".png"), dpi=300)


COLOR_SIMILARITY_FACTOR = 0.5

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='People Segmentation')
    argparser.add_argument('-m', '--model_name', type=str, help='Name of segmentation model',
                           default="xception65_coco_voc_trainval")
    argparser.add_argument('-i', '--images_dir', type=str, help='Path to images files', default="images")
    argparser.add_argument('-o', '--output_dir', type=str, help='Path to output folder', default="output")
    argparser.add_argument('-b', '--bg_blend', type=str, help='OPTIONAL: bg image to blend')
    argparser.add_argument('-d', '--plot_debug', type=int, default=0)
    argparser.add_argument('-s', '--output_size', type=int, default=2000)
    args = argparser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot_debug == 1:
        os.makedirs(os.path.join(args.output_dir, "debug"), exist_ok=True)

    if 'xception65' in args.model_name:
        model = DeepLab(args.model_name)
    else:
        model = Gluon(args.model_name)

    for filename in os.listdir(args.images_dir):
        print(f"processing img: {filename}")
        img = cv2.imread(os.path.join(args.images_dir, filename))
        if img is None:
            continue

        scale = args.output_size / np.max(img.shape[:-1])
        width = int((img.shape[1] * scale / 2)) * 2
        height = int((img.shape[0] * scale / 2)) * 2
        img = cv2.resize(img, (width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)

        pred, small_img, padding, resize_properties = model.eval_pepole_and_animal(img)
        pred_th, pred_hesitant = process_pred(pred, padding, model.get_props()['th'])
        pred_th = remove_small_connected_componnets(pred_th)
        # grads = get_grad(small_img)
        # pred_th_clean = clear_mask_by_grads(pred_th, grads)
        pred_th_clean = pred_th

        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(pred)
        # axs[0, 1].imshow(pred_th)
        # axs[1, 0].imshow(pred_th_clean)
        # axs[1, 1].imshow(grads)
        # plt.show()

        small_img = small_img / 255.
        img = img / 255.

        fg_centers, fg_centers_probs = get_charactistic_colors(small_img[np.squeeze(pred_th_clean == 1)])
        bg_centers, bg_centers_probs = get_charactistic_colors(
            small_img[np.logical_and(np.squeeze(pred_th_clean == 0), np.all(padding == 1, axis=2))])

        pred_hesitant_os = back_to_original_size(pred_hesitant, resize_properties)
        fg_sim = get_similarity(img, fg_centers, fg_centers_probs) * pred_hesitant_os
        bg_sim = get_similarity(img, bg_centers, bg_centers_probs) * pred_hesitant_os
        bg_fg_sim_diff = fg_sim - bg_sim

        mask = get_mask(back_to_original_size(pred_th_clean, resize_properties), bg_fg_sim_diff,
                        COLOR_SIMILARITY_FACTOR)
        mask = remove_small_connected_componnets(mask)

        bg = get_bg(args.bg_blend, img.shape[:-1][::-1])
        blend = blend_img_bg(img, bg, np.expand_dims(cv2.blur(mask, (25, 25), 0), 2), num_levels=1)
        cv2.imwrite(os.path.join(args.output_dir, filename + "_" + args.bg_blend.split('/')[-1] + ".png"),
                    cv2.cvtColor((np.clip(blend, 0., 1.) * 255.).astype(np.uint8), cv2.COLOR_BGRA2RGBA))

        if args.plot_debug == 1:
            plot_debug(img, bg_centers, fg_centers, pred, pred_th, bg_fg_sim_diff, mask, blend, pred_th_clean)
