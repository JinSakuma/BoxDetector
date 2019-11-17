# --------------------------------------------------------------
# CPN: ResNet50 config
# by Longqi-S
# --------------------------------------------------------------

import numpy as np
from models.config import DefaultConfig


class Config(DefaultConfig):
    NAME = "ResNet50_CPN"  # Override in sub-classes
    GPUs = '0'   # '0, 1, 2, 3, 4, 5, 6, 7'
    IMAGES_PER_GPU = 16
    STEPS_PER_EPOCH = 10    # 4000
    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 2
    KEYPOINTS_NUM = 7
    LEARNING_RATE = 0.01
    LEARNING_MOMENTUM = 0.9
    # Weight decay regularization
    WEIGHT_DECAY = 1e-5
    LEARNING_OPTIMIZER = 'SGD'   # 'SGD', 'ADAM'
    GRADIENT_CLIP_NORM = 5.0
    #########################################################################
    # preprocessing configuration
    #########################################################################
    IMG_PATH = "../CPN/images0829/"
    DATA_SHAPE = (512, 512)  # height, width
    OUTPUT_SHAPE = (128, 128)  # height, width
    GAUSSAIN_KERNEL = (13, 13)
    #
    GK15 = (23, 23)
    GK11 = (17, 17)
    GK9 = (13, 13)
    GK7 = (9, 9)

    DATA_AUG = True  # has to be true
    NR_AUG = 2
    PIXEL_MEANS_VARS = False
    PIXEL_MEANS = np.array([[[103.939, 116.779, 123.68]]])   # BGR
    PIXEL_NORM = False
    imgExtXBorder = 0.1
    imgExtYBorder = 0.15
    symmetry = [(1, 3), (4, 5)]

    #########################################################################
    # network configuration
    #########################################################################
    BACKBONE = 'resnet50'
