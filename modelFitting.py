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

from lines import tennis_court_model_points, tennis_court_model_lines

from sklearn.mixture import GaussianMixture

### VISUAL DEBUG ###
# print(np.amax(tennis_court_model_points, axis=0).shape)
# img_with_projected_lines = np.zeros(np.append(np.amax(tennis_court_model_points, axis=0)[[1,0]], [3]))
# for line in tennis_court_model_lines:
#    img_with_projected_lines = cv2.line(img_with_projected_lines, tennis_court_model_points[line[0]], tennis_court_model_points[line[1]], (0, 255, 0), thickness=2)

# cv2.imshow('model', img_with_projected_lines)
# cv2.waitKey(0)

### END VISUAL DEBUG ###

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

def test_single_image(cfg, impath, model, device, output_path = "", threshold = 0.97):
    image = cv2.imread(impath)
    lines = get_lines_from_nn(cfg, impath, image[:, :, [2, 1, 0]], model, device, threshold)
    print('number of lines: ', len(lines))

    ### VISUAL DEBUG ###
    # img_with_lines = np.copy(image)
    # for line in lines:
    #     line = line.astype(np.int32)
    #     img_with_lines = cv2.line(img_with_lines, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)
    #     img_with_lines = cv2.circle(img_with_lines, (line[0], line[1]), 2, (255, 80, 0), 3)
    #     img_with_lines = cv2.circle(img_with_lines, (line[2], line[3]), 2, (255, 80, 0), 3)
    # cv2.imshow('img_with_lines', img_with_lines)
    # cv2.waitKey(0)
    ### END VISUAL DEBUG ###

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
    for line in lines:
        line = line.astype(np.int32)
        mask = cv2.line(mask, (line[0], line[1]), (line[2], line[3]), 255, 8)
    
    ### VISUAL DEBUG ###
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    ### END VISUAL DEBUG ###

    mask = mask.astype(bool)
    color_list = image[mask]
    print(color_list.shape)
    gm = GaussianMixture(n_components=3, random_state=0).fit(color_list)
    line_gaussian = gm.predict([[255, 255, 255]])[0]
    
    flatten_color = image.reshape(-1, image.shape[-1])
    fitted_gaussian = gm.predict(flatten_color)

    candidate_lines = np.logical_and(mask, np.reshape(fitted_gaussian == line_gaussian, (image.shape[0], image.shape[1])))
    points = np.asarray([np.append(lines[:,0], lines[:,2]),np.append(lines[:,1], lines[:,3])]).T

    # print(points.shape)
    # print(tennis_court_model_points.shape)

    ### VISUAL DEBUG ###
    # img_with_points = np.copy(image)
    # for point in points:
    #     point = point.astype(np.int32)
    #     img_with_points = cv2.circle(img_with_points, (point[0], point[1]), 2, (255, 80, 0), 3)
    # img_with_lines = np.copy(image)
    # for line in lines:
    #     line = line.astype(np.int32)
    #     img_with_lines = cv2.line(img_with_lines, line[0:2], line[2:4], (255, 80, 0), thickness = 2)
    # cv2.imshow('img_with_lines', img_with_lines)
    # cv2.waitKey(0)
    ### END VISUAL DEBUG ###

    tennis_court_model_points_reshaped = np.float32(tennis_court_model_points[:, np.newaxis, :])
    points_reshaped = np.float32(points[:, np.newaxis, :])

    # print(points.T.shape)
    points_to_project = np.r_[points.T, np.full((1, points.shape[0]), 1, dtype=points.dtype)]

    best_RT_matrix = None
    best_score = sys.float_info.max
    best_fitting_points = []
    best_projected_points = None

    ### VISUAL DEBUG ###
    # original_points = np.asarray([[134,361],[662,369],[651,235],[292,235]])
    # tennis_court_point_idx = np.asarray([4,5,7,6])
    # 
    # print(np.amax(tennis_court_model_points, axis=0).shape)
    # img_with_projected_lines = np.zeros(np.append(np.amax(tennis_court_model_points, axis=0)[[1,0]], [3]))
    # for idx in tennis_court_point_idx:
    #    img_with_projected_lines = cv2.circle(img_with_projected_lines, tennis_court_model_points[idx].astype(np.int32), 4, (255, 0, 0), thickness=-1)
    # 
    # cv2.imshow('model', img_with_projected_lines)
    # cv2.waitKey(0)
    # 
    # RT_matrix, mask = cv2.findHomography(tennis_court_model_points[tennis_court_point_idx].astype(np.float32)[:, np.newaxis, :], original_points.astype(np.float32)[:, np.newaxis, :])
    # tennis_court_projected_points = RT_matrix @ np.r_[tennis_court_model_points.T, np.full((1, tennis_court_model_points.shape[0]), 1.0, dtype=np.float32)]
    # tennis_court_projected_points = tennis_court_projected_points / tennis_court_projected_points[2]
    # tennis_court_projected_points = tennis_court_projected_points.T
    # img_with_projected_lines = np.copy(image)
    # for line in tennis_court_model_lines:
    #     img_with_projected_lines = cv2.line(img_with_projected_lines, tennis_court_projected_points[line[0]][0:2].astype(np.int32), tennis_court_projected_points[line[1]][0:2].astype(np.int32), (255, 0, 0), thickness=2)
    # for model_point in tennis_court_point_idx:
    #     img_with_projected_lines = cv2.circle(img_with_projected_lines, tennis_court_projected_points[model_point][0:2].astype(np.int32), 4, (255, 0, 0), thickness=-1)
    # for original_point in original_points:
    #     img_with_projected_lines = cv2.circle(img_with_projected_lines, original_point.astype(np.int32), 2, (0, 255, 0), thickness=-1)
    # cv2.imshow('img_with_projected_lines', img_with_projected_lines)
    # cv2.waitKey(0)
    
    # print("best_rtmse: ", best_rtmse)
    # img_wrap = cv2.warpPerspective(image, np.linalg.inv(RT_matrix), (144, 312))
    # cv2.imshow('img_wrap', img_wrap)
    # cv2.waitKey(0)
    ### END VISUAL DEBUG ###

    model_image = np.zeros(np.amax(tennis_court_model_points, axis=0)[[1,0]] + 1)
    for line in tennis_court_model_lines:
        model_image = cv2.line(model_image, tennis_court_model_points[line[0]], tennis_court_model_points[line[1]], (255), thickness=2)

    for i in range(100000):
        select_lines_idx = np.random.choice(lines.shape[0], size=(9,), replace=False)
        # select_points_idx = np.random.choice(points.shape[0], size=(4,), replace=False) # scegliere in base a linee casuali invece di punti casuali
        select_model_lines_idx = np.random.choice(tennis_court_model_lines.shape[0], size=(6,), replace=False)
        # model_points_idx = np.random.choice(tennis_court_model_points.shape[0], size=(4,), replace=False)

        select_points = np.asarray([np.append(lines[select_lines_idx,0], lines[select_lines_idx,2]),np.append(lines[select_lines_idx,1], lines[select_lines_idx,3])]).T
        # print("select_points")
        # print(select_points)
        # select_points = points[select_points_idx]
        
        # select_model_points = tennis_court_model_points[np.append(tennis_court_model_lines[select_model_lines_idx,0], tennis_court_model_lines[select_model_lines_idx,1])]
        select_model_points = tennis_court_model_points[np.append(tennis_court_model_lines[:,0], tennis_court_model_lines[:,1])]
        # print("select_model_points")
        # print(select_model_points)
        # select_model_points = tennis_court_model_points[model_points_idx]

        # print("homography points:")
        # print(points[select_points_idx].astype(np.float32))
        # print("Tennis court model points:")
        # print(tennis_court_model_points[model_points_idx].astype(np.float32))
        RT_matrix, mask = cv2.findHomography(select_model_points.astype(np.float32)[:, np.newaxis, :], select_points.astype(np.float32)[:, np.newaxis, :])
        # print("mask:")
        # print(mask)
        # print("RT_matrix:")
        # print(RT_matrix)
        if np.sum(np.isinf(RT_matrix)) != 0:
            continue
        if np.sum(mask) != 4:
            continue
        # img_wrap = cv2.warpPerspective(image, RT_matrix, (78, 36))
        # print(img_wrap.shape)
        # cv2.imshow('img_wrap', img_wrap)
        # cv2.waitKey(0)

        tennis_court_projected_points = RT_matrix @ np.r_[tennis_court_model_points.T, np.full((1, tennis_court_model_points.shape[0]), 1, dtype=np.float32)]
        tennis_court_projected_points = tennis_court_projected_points / tennis_court_projected_points[2]
        tennis_court_projected_points = tennis_court_projected_points.T
        
        # print(tennis_court_projected_points)
        ### VISUAL DEBUG ###
        img_with_projected_lines = np.copy(image)
        for line in tennis_court_model_lines:
            img_with_projected_lines = cv2.line(img_with_projected_lines, tennis_court_projected_points[line[0]][0:2].astype(np.int32), tennis_court_projected_points[line[1]][0:2].astype(np.int32), (255, 0, 0), thickness=2)
        for model_point in select_model_points:
            img_with_projected_lines = cv2.circle(img_with_projected_lines, model_point[0:2].astype(np.int32), 4, (255, 0, 0), thickness=-1)
        for select_point in select_points:
            img_with_projected_lines = cv2.circle(img_with_projected_lines, select_point.astype(np.int32), 2, (0, 255, 0), thickness=-1)
        cv2.imshow('img_with_projected_lines', img_with_projected_lines)
        # cv2.waitKey(0)
        ### END DEBUG ###

        # rtmse = 0.0
        # fitting_points = []
        # for point in tennis_court_projected_points:
        #     distances = np.sum(np.square(points[:,0:2] - point[0:2]), axis=1)
        #     min_point = np.argmin(distances)
        #     fitting_points.append(min_point)
        #     rtmse += distances[min_point]

        mask_with_projected_lines = np.zeros(image.shape[:2], np.uint8)
        for line in tennis_court_model_lines:
            mask_with_projected_lines = cv2.line(mask_with_projected_lines, tennis_court_projected_points[line[0]][0:2].astype(np.int32), tennis_court_projected_points[line[1]][0:2].astype(np.int32), (255), thickness=2)
        colors_to_predict = image[mask_with_projected_lines.astype(bool)]
        best_gaussian = gm.predict(colors_to_predict)
        score = np.sum(best_gaussian == line_gaussian)
        # masked_image = cv2.bitwise_and(image, image, mask=mask_with_projected_lines)
        # masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        # masked_image[np.logical_and(masked_image > 0,masked_image < 200)] = -300
        # score = masked_image.sum()
        
        # print(score)
        
        """
        img = np.copy(image)
        for point in fitting_points:
            img = cv2.circle(img, points[point].astype(np.int32), 5, (255, 0, 0), -1)
        cv2.imshow('window', img)
        cv2.waitKey(0)
        """

        if abs(np.linalg.det(RT_matrix)) < sys.float_info.epsilon:
            continue

        inverse_matrix = np.linalg.inv(RT_matrix)

        tennis_court_reprojected_points = inverse_matrix @ tennis_court_projected_points.T
        tennis_court_reprojected_points = tennis_court_reprojected_points / tennis_court_reprojected_points[2]
        tennis_court_reprojected_points = tennis_court_reprojected_points.T

        ### VISUAL DEBUG ###
        print("RT_matrix:")
        print(RT_matrix)
        img_with_projected_lines = np.zeros(np.amax(tennis_court_model_points, axis=0)[[1,0]])
        for line in tennis_court_model_lines:
           img_with_projected_lines = cv2.line(img_with_projected_lines, tennis_court_reprojected_points[line[0]][0:2].astype(np.int32), tennis_court_reprojected_points[line[1]][0:2].astype(np.int32), (255), thickness=2)
        cv2.imshow('model', img_with_projected_lines)
        # cv2.waitKey(0)
        ### END VISUAL DEBUG ###

        img_wrap = cv2.warpPerspective(image, np.linalg.inv(RT_matrix), (model_image.shape[1], model_image.shape[0]))
        img_wrap_gray = cv2.cvtColor(img_wrap, cv2.COLOR_BGR2GRAY)

        ### VISUAL DEBUG ###
        cv2.imshow('img_wrap_gray', img_wrap_gray)
        cv2.waitKey(0)
        ### END VISUAL DEBUG ###

        score = np.sum(np.square(img_wrap_gray - model_image))

        if best_score > score:
            best_score = score
            best_RT_matrix = RT_matrix
            best_fitting_points = select_points
            best_projected_points = tennis_court_projected_points
    
    best_fitting_points = np.asarray(best_fitting_points)

    print("best_score:", best_score)
    img_with_projected_lines = np.copy(image)
    for line in tennis_court_model_lines:
        img_with_projected_lines = cv2.line(img_with_projected_lines, best_projected_points[line[0]][0:2].astype(np.int32), best_projected_points[line[1]][0:2].astype(np.int32), (255, 0, 0), thickness=2)
    cv2.imshow('window', img_with_projected_lines)
    cv2.waitKey(0)
    
    # print("best_rtmse: ", best_rtmse)
    # img_wrap = cv2.warpPerspective(image, np.linalg.inv(RT_matrix), (144, 312))
    # cv2.imshow('img_wrap', img_wrap)
    # cv2.waitKey(0)

        

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
        test_single_image(cfg, os.path.join(args.img_directory, args.img), model, device, output_path = output_path, threshold = args.threshold)

if __name__ == "__main__":
    args = argument_parsing()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger('hawp', output_dir)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    test(cfg, args)