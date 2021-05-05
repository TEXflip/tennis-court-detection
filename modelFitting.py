import torch
import hawp.parsing
from hawp.parsing.config import cfg
from hawp.parsing.utils.comm import to_device
from hawp.parsing.dataset.build import build_transform
from hawp.parsing.detector import WireframeDetector
from hawp.parsing.utils.logger import setup_logger
from hawp.parsing.utils.metric_logger import MetricLogger
from hawp.parsing.utils.miscellaneous import save_config
from hawp.parsing.utils.checkpoint import DetectronCheckpointer
from skimage import io
import os
import os.path as osp
import time
import datetime
import argparse
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

import numpy as np
import cv2
import sys
import random

from lines import tennis_court_model_points

def argument_parsing():
    parser = argparse.ArgumentParser(description='HAWP Testing')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        required=True,
                        )

    parser.add_argument("--img",default="",type=str,required=False,
                        help="image path")

    parser.add_argument("--img_directory",default="",type=str,required=False,
                        help="input images directory")
    parser.add_argument("--output_path",type=str,required=False,
                        help="output path, img not show if specified")

    parser.add_argument("--threshold",
                        type=float,
                        default=0.97)
    
    return parser.parse_args()

def get_lines_from_nn(cfg, impath, image, model, device, threshold):
    transform = build_transform(cfg)
    image_tensor = transform(image.astype(float))[None].to(device)
    meta = {
        'filename': impath,
        'height': image.shape[0],
        'width': image.shape[1],
    }

    with torch.no_grad():
        output, _ = model(image_tensor,[meta])
        output = to_device(output,'cpu')
    
    lines = output['lines_pred'].numpy()
    scores = output['lines_score'].numpy() # possible use for matching priority
    idx = scores>threshold

    return lines[idx]

def computeC2MC1(r1, tvec1, r2, tvec2, )

def test_single_image(cfg, impath, model, device, output_path = "", threshold = 0.97):
    image = cv2.imread(impath)
    lines = get_lines_from_nn(cfg, impath, image[:, :, [2, 1, 0]], model, device, threshold)
    points = np.float32(np.asarray([np.append(lines[:,0], lines[:, 1]), np.append(lines[:, 2], lines[:, 3])]).T)
    print(points.shape)
    print(tennis_court_model_points.shape)
    tennis_court_model_points_reshaped = np.float32(tennis_court_model_points[:, np.newaxis, :])
    points_reshaped = np.float32(points[:, np.newaxis, :])

    """
    MIN_MATCH_COUNT = 10
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image,None)
    # kp2, des2 = sift.detectAndCompute(img2,None)
    print(des1.shape)
    print(kp1)
    """

    for i in range(1000):
        np.random_sample()
        points_to_test = points[random.sample(points.shape[0], size=(4,), replace=False)]
        
        # tennis_court_to_test = tennis_court_model_points_reshaped[random.sample(points.shape[0], size=(4,), replace=False)]
        tennis_court_to_test = tennis_court_model_points_reshaped[[0,3,4,5]]
        
        retval, mask = cv2.findHomography(points_to_test, tennis_court_to_test, cv2.RANSAC, ransacReprojThreshold = 10)
        projected_points = mask @ tennis_court_model_points_reshaped
        projected_points

def model_loading(cfg):
    logger = logging.getLogger("hawp.testing")
    device = cfg.MODEL.DEVICE
    model = WireframeDetector(cfg)
    model = model.to(device)

    checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)
    _ = checkpointer.load()
    model = model.eval()
    return model, device

def test(cfg, args):
    model, device = model_loading(cfg)

    if args.img == "":
        if args.img_directory == "":
            print("Image or image directory must be specify")
            sys.exit(1)
        base_output_path = ""
        if args.output_path != "":
            os.makedirs(args.output_path, exist_ok=True)
            base_output_path = args.output_path
        for impath in os.listdir(args.img_directory):
            print("Predicting image ", os.path.join(args.img_directory,impath))
            if impath.endswith('.jpg') or impath.endswith('.jpeg'):
                output_path = ""
                if base_output_path != "":
                    output_path = os.path.join(base_output_path, impath)
                test_single_image(cfg, os.path.join(args.img_directory, impath), model, device, output_path = output_path, threshold = args.threshold)
    else:
        output_path = ""
        if args.output_path != "":
            output_path = args.output_path
        test_single_image(cfg, os.path.join(args.img_directory, impath), model, device, output_path = output_path, threshold = args.threshold)

if __name__ == "__main__":
    args = argument_parsing()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger('hawp', output_dir)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    test(cfg, args)