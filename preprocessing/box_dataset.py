# --------------------------------------------------------------
# CPN: dataset preprocessing
# by Longqi-S
# --------------------------------------------------------------
import numpy as np
import random
import cv2


def data_augmentation(trainData, trainLabel, cfg):
    """
    1) オリジナル
    2) 反転
    3) 縮小
    4) 回転
    """
    tremNum = cfg.NR_AUG - 1
    gotData = trainData.copy()
    trainData = np.append(trainData, [trainData[0] for i in range(tremNum * len(trainData))], axis=0)
    trainLabel = np.append(trainLabel, [trainLabel[0] for i in range(tremNum * len(trainLabel))], axis=0)

    counter = len(gotData)
    for lab in range(len(gotData)):
        ori_img = gotData[lab]
        annot = trainLabel[lab].copy()
        height, width = ori_img.shape[:2]
        center = (width / 2., height / 2.)
        keypoints_num = cfg.KEYPOINTS_NUM

        trainData[lab] = ori_img
        trainLabel[lab] = annot

        if tremNum > 0:
            # flip augmentation
            newimg = cv2.flip(ori_img, 1)
            cod = []
            allc = []
            for i in range(keypoints_num):
                x, y = annot[i << 1], annot[i << 1 | 1]
                if y >= 0:
                    y = width - 1 - y
                cod.append((x, y))
            trainData[counter] = newimg
            # 左右対称なKeypointの反転
            for (q, w) in cfg.symmetry:
                cod[q], cod[w] = cod[w], cod[q]
            for i in range(keypoints_num):
                allc.append(cod[i][0])
                allc.append(cod[i][1])
            trainLabel[counter] = np.array(allc)
            counter += 1

        if tremNum > 1:
            # shrink augmentation
            size = 768
            tmp = ori_img[:, :]
            start = int((size - height) / 2)
            fin = int((size + height) / 2)
            new_img = cv2.resize(np.ones((1, 1, 3), np.uint8)*255, (size, size))
            new_img[start:fin, start:fin] = tmp
            resized = cv2.resize(new_img, (512, 512))
            trainData[counter] = resized
            trainLabel[counter] = (annot+128)/1.5
            counter += 1

        if tremNum > 2:
            # rotated augmentation
            angle = random.uniform(10, 20)
            if random.randint(0, 1):
                angle *= -1
            rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
            newimg = cv2.warpAffine(resized, rotMat, (width, height))
            allc = []

            for i in range(keypoints_num):
                x, y = trainLabel[counter-1][i << 1], trainLabel[counter-1][i << 1 | 1]
                coor = np.array([y, x])
                if x >= 0 and y >= 0:
                    R = rotMat[:, : 2]
                    W = np.array([rotMat[0][2], rotMat[1][2]])
                    coor = np.dot(R, coor) + W
                allc.append(coor[1])
                allc.append(coor[0])

            trainData[counter] = newimg
            trainLabel[counter] = np.array(allc)

        return trainData, trainLabel


def joints_heatmap_gen(data, label, cfg, return_valid=False, gaussian_kernel=(13, 13)):
    """
     generate corresponding heatmaps
    """
    num_keypoints = cfg.KEYPOINTS_NUM
    tar_size = cfg.OUTPUT_SHAPE
    ori_size = cfg.DATA_SHAPE
    bat_size = len(data)
    if return_valid:
        valid = np.ones((bat_size, num_keypoints), dtype=np.float32)
    ret = np.zeros((bat_size, num_keypoints, tar_size[0], tar_size[1]), dtype='float32')
    for i in range(bat_size):
        for j in range(num_keypoints):
            idx_x = j*2
            idx_y = idx_x+1
            if label[i][idx_x] < 0 or label[i][idx_y] < 0:
                continue
            label[i][idx_y] = min(label[i][idx_y], ori_size[0] - 1)
            label[i][idx_x] = min(label[i][idx_x], ori_size[1] - 1)
            ret[i][j][int(label[i][idx_x] * tar_size[0] / ori_size[0])][int(label[i][idx_y] * tar_size[1] / ori_size[1])] = 1

    for i in range(bat_size):
        for j in range(num_keypoints):
            ret[i, j] = cv2.GaussianBlur(ret[i, j], gaussian_kernel, 0)

    for i in range(bat_size):
        for j in range(num_keypoints):
            am = np.amax(ret[i][j])
            if am <= 1e-8:
                if return_valid:
                    valid[i][j] = 0.
                continue
            ret[i][j] /= am / 255
    if return_valid:
        return ret, valid
    else:
        return ret


def _preprocess_zero_mean_unit_range(inputs):
    """Map image values from [0, 255] to [-1, 1]."""
    return (2.0 / 255.0) * inputs.astype(np.float32) - 1.0


def image_preprocessing(inputs, config):
    """ image preprocessing
    resnet from keras: just original img
    resnet from tensorflow: 1) sub mean; 2) div 255;
    """
    img = inputs.astype(np.float32)
    if config.PIXEL_MEANS_VARS:
        img = img - config.PIXEL_MEANS
        if config.PIXEL_NORM:
            img = img / 255.
    return img


def preprocessing(d, config, stage='train'):
    """ preprocessing method
    1) read imgs and crop based on bbox;
    2) data augment;
    3) generate heatmaps;
    """
    height, width = config.DATA_SHAPE
    imgs = []
    labels = []
    valids = []
    # read image
    img_ori = cv2.imread(d['path'])
    if img_ori is None:
        print('read none image')
        return None
    # crop based on bbox
    img = img_ori.copy()

    joints = d[1:].values.reshape((config.KEYPOINTS_NUM, 2)).astype(np.float32)
    label = joints[:, :2].copy()
    # 全てvisible
    valid = np.array([2, 2, 2, 2, 2, 2, 2])

    labels.append(label.reshape(-1))
    valids.append(valid.reshape(-1))
    imgs.append(img)
    if stage == 'train':
        imgs, labels = data_augmentation(imgs, labels, config)
        while len(valids) > len(imgs):
            valids.append(valid.reshape(-1))

        heatmaps15 = joints_heatmap_gen(imgs, labels, config, return_valid=False, gaussian_kernel=config.GK15)
        heatmaps11 = joints_heatmap_gen(imgs, labels, config, return_valid=False, gaussian_kernel=config.GK11)
        heatmaps9 = joints_heatmap_gen(imgs, labels, config, return_valid=False, gaussian_kernel=config.GK9)
        heatmaps7 = joints_heatmap_gen(imgs, labels, config, return_valid=False, gaussian_kernel=config.GK7)
        imgs = np.asarray(imgs).astype(np.float32)
        valids = np.asarray(valids)
        for index_ in range(len(imgs)):
            imgs[index_] = image_preprocessing(imgs[index_], config)

        return_args = [imgs.astype(np.float32),
                       heatmaps15.astype(np.float32).transpose(0, 2, 3, 1),
                       heatmaps11.astype(np.float32).transpose(0, 2, 3, 1),
                       heatmaps9.astype(np.float32).transpose(0, 2, 3, 1),
                       heatmaps7.astype(np.float32).transpose(0, 2, 3, 1),
                       valids.astype(np.float32)
                       ]
        return return_args
    else:
        for index_ in range(len(imgs)):
            imgs[index_] = image_preprocessing(imgs[index_], config)
        return [np.asarray(imgs).astype(np.float32)]
