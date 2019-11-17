import os
import numpy as np
import configs
import argparse
import cv2
import time
import configs
import keras.backend as K
from tqdm import tqdm
from preprocessing.box_dataset import preprocessing
from models import cpn as modellib


def save_result(img, landmark, path):
    h, w, c = img.shape
    n = int(max(h, w) / 128)

    copy = img.copy()
    copy2 = img.copy()
    for y, x in landmark:
        copy[int(y)-n:int(y)+n, int(x)-n:int(x)+n] = [0, 0, 255]

    landmark = landmark.astype("int64")
    cv2.line(copy2, (landmark[0][1], landmark[0][0]), (landmark[1][1], landmark[1][0]), (0, 255, 0), 3)
    cv2.line(copy2, (landmark[0][1], landmark[0][0]), (landmark[2][1], landmark[2][0]), (0, 255, 0), 3)
    cv2.line(copy2, (landmark[0][1], landmark[0][0]), (landmark[3][1], landmark[3][0]), (0, 255, 0), 3)
    cv2.line(copy2, (landmark[1][1], landmark[1][0]), (landmark[4][1], landmark[4][0]), (0, 255, 0), 3)
    cv2.line(copy2, (landmark[2][1], landmark[2][0]), (landmark[5][1], landmark[5][0]), (0, 255, 0), 3)
    cv2.line(copy2, (landmark[4][1], landmark[4][0]), (landmark[2][1], landmark[2][0]), (0, 255, 0), 3)
    cv2.line(copy2, (landmark[3][1], landmark[3][0]), (landmark[5][1], landmark[5][0]), (0, 255, 0), 3)
    cv2.line(copy2, (landmark[6][1], landmark[6][0]), (landmark[1][1], landmark[1][0]), (0, 255, 0), 3)
    cv2.line(copy2, (landmark[6][1], landmark[6][0]), (landmark[3][1], landmark[3][0]), (0, 255, 0), 3)

    name = path.split(".")
    path1 = "."+name[1]+"_plot."+name[2]
    path2 = "."+name[1]+"_grid."+name[2]
    print(path1, path2)
    cv2.imwrite(path1, copy)
    cv2.imwrite(path2, copy2)


def resize(img):
    tmp = img[:, :]
    height, width = img.shape[:2]
    if(height > width):
        size = height
        limit = width
    else:
        size = width
        limit = height
    start = int((size - limit) / 2)
    fin = int((size + limit) / 2)
    new_img = cv2.resize(np.zeros((1, 1, 3), np.uint8), (size, size))
    if(size == height):
        new_img[:, start:fin] = tmp
    else:
        new_img[start:fin, :] = tmp

    return cv2.resize(new_img, (512, 512))


def reproduce(img_o, landmarks):
    h, w, c = img_o.shape
    if h > w:
        size = h
        add = (size - w) / 2
    else:
        size = w
        add = (size - h) / 2

    ori_land = landmarks * (size / 512)
    if h > w:
        ori_land[:, 1] -= add
    else:
        ori_land[:, 0] -= add

    return ori_land


def test(model, img, resized, config, save_path):
    test_img = resized[np.newaxis, ...]
    # 左右対象の組を作る
    test_imgs = []
    test_imgs.append(test_img)
    feed = test_imgs
    ori_img = test_imgs[0][0]
    flip_img = cv2.flip(ori_img, 1)
    feed.append(flip_img[np.newaxis, ...])
    feed = np.vstack(feed)
    # model predict
    res = model.keras_model.predict([feed], verbose=0)
    res = res.transpose(0, 3, 1, 2)     # [batch, kps, h, w]

    # combine flip result
    ii_ = 0
    fmp = res[1 + ii_].transpose((1, 2, 0))
    fmp = cv2.flip(fmp, 1)
    fmp = list(fmp.transpose((2, 0, 1)))
    for (q, w) in config.symmetry:
        fmp[q], fmp[w] = fmp[w], fmp[q]
    fmp = np.array(fmp)
    res[ii_] += fmp
    res[ii_] /= 2

    cls_skeleton = np.zeros((1, config.KEYPOINTS_NUM, 3))

    test_image_id = 0
    start_id = 0
    r0 = res[test_image_id - start_id].copy()
    r0 /= 255
    r0 += 0.5
    for w in range(config.KEYPOINTS_NUM):
        res[test_image_id - start_id, w] /= np.amax(res[test_image_id - start_id, w])

    border = 10
    dr = np.zeros((config.KEYPOINTS_NUM, config.OUTPUT_SHAPE[0] + 2 * border, config.OUTPUT_SHAPE[1] + 2 * border))
    dr[:, border:-border, border:-border] = res[test_image_id - start_id][: config.KEYPOINTS_NUM].copy()

    for w in range(config.KEYPOINTS_NUM):
        dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)

    # find first max and second max one
    for w in range(config.KEYPOINTS_NUM):
        lb = dr[w].argmax()
        y, x = np.unravel_index(lb, dr[w].shape)
        dr[w, y, x] = 0
        lb = dr[w].argmax()
        py, px = np.unravel_index(lb, dr[w].shape)
        y -= border
        x -= border
        py -= border + y
        px -= border + x
        ln = (px ** 2 + py ** 2) ** 0.5
        delta = 0.25
        if ln > 1e-3:
            x += delta * px / ln
            y += delta * py / ln
        x = max(0, min(x, config.OUTPUT_SHAPE[1] - 1))
        y = max(0, min(y, config.OUTPUT_SHAPE[0] - 1))
        cls_skeleton[test_image_id, w, :2] = (y * 4 + 2, x * 4 + 2)
        cls_skeleton[test_image_id, w, 2] = r0[w, int(round(x) + 1e-10), int(round(y) + 1e-10)]

    landmarks = cls_skeleton[:, :, :2][0]
    ori_land = reproduce(img, landmarks)
    save_result(img, ori_land, save_path)

    return landmarks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-d', type=str, dest='gpu_id', default='0')
    parser.add_argument('--model', '-m', type=str, dest='test_model', default="./data/model/cpn_resnet50_cpn_0039.h5")
    parser.add_argument('--cfg', '-c', type=str, dest='cfg', default="./configs/BOX_CPN_ResNet50_FPN_cfg.py")
    parser.add_argument('--save_dir', '-s', required=True, type=str, dest='save_path')
    parser.add_argument('--img_dir', '-i', required=True, type=str, dest='img_folder')
    args = parser.parse_args()

    config_file = os.path.basename(args.cfg).split('.')[0]
    config_def = eval('configs.' + config_file + '.Config')
    config = config_def()
    config.GPUs = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUs

    model = modellib.CPN(mode="inference", config=config, model_dir="")
    model.load_weights(args.test_model, by_name=True)

    imgs_path = args.img_folder
    img_names = os.listdir(imgs_path)
    for name in img_names:
        path = imgs_path + "/" + name
        img = cv2.imread(path)
        resized = resize(img)
        test(model, img, resized, config, args.save_path+"/"+name)

    print('finished!!')
