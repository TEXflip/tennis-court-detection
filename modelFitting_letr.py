from letr_inference import LETRInference
import os
import argparse

import numpy as np
import cv2
import sys
import torch
import networkx as nx
from shapely.geometry import LineString
torch.cuda.is_available = lambda : False

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

    parser.add_argument("--checkpoint-filepath",
                        metavar="FILE",
                        help="path to checkpoint file",
                        type=str,
                        required=True,
                        )

    parser.add_argument("--img",default="",type=str,required=False,
                        help="image path")

    parser.add_argument("--img_directory",default="",type=str,required=False,
                        help="input images directory")
    parser.add_argument("--output_path",type=str,required=False,
                        help="output path, img not show if specified")
    parser.add_argument("--threshold-letr-score",
                        type=float,
                        default=0.5)
    parser.add_argument("--threshold",
                        type=float,
                        default=0.97)
    
    return parser.parse_args()

def pointLineMinDist(line, point): # (a.x,a.y,b.x,b.y), (p.x,p.y)
    ap = point - line[0:2]
    ab = line[2:4] - line[0:2]
    perpendicular_intersection = line[0:2] + max(0, min(1, np.dot(ap, ab)/(ab**2).sum())) * ab
    return np.linalg.norm(perpendicular_intersection - point)

def linesFiltering(lines, imgRes, angleTh = 5, distTh = 10, minLength = 0.1):
    out = []
    minRes = min(imgRes)
    for i, line1 in enumerate(lines):
        append = True
        v1 = line1[2:4] - line1[0:2]
        len1 = np.linalg.norm(v1)
        if len1 < minRes * minLength:
            continue

        for j, line2 in enumerate(lines):
            if i == j:
                continue
            v2 = line2[2:4] - line2[0:2]
            len2 = np.linalg.norm(v2)

            if len2 < minRes * minLength:
                continue

            dot = np.dot(v1 / len1, v2 / len2)
            dot = max(-1, min(dot, 1))
            angle = np.arccos(dot) * 180 / np.pi

            angleCondition = np.abs(angle) < angleTh or (angle > 180 - angleTh and angle < 180 + angleTh)

            dist1 = np.linalg.norm(line1[0:2]-line2[0:2]) < distTh
            dist2 = np.linalg.norm(line1[0:2]-line2[2:4]) < distTh
            dist3 = np.linalg.norm(line1[2:4]-line2[0:2]) < distTh
            dist4 = np.linalg.norm(line1[2:4]-line2[2:4]) < distTh
            distCondition = dist1 or dist2 or dist3 or dist4
            # dist1 = pointLineMinDist(line1, line2[0:2])
            # dist2 = pointLineMinDist(line1, line2[2:4])
            # dist3 = pointLineMinDist(line2, line1[0:2])
            # dist4 = pointLineMinDist(line2, line1[2:4])
            # distCondition = min((dist1, dist2, dist3, dist4)) < distTh

            if angleCondition and distCondition and len1 < len2:
                append = False
                break

        if append:
            out.append(line1)

    return np.asarray(out)

def linesFilteringWithGraph(lines, min_components = 3, lineExtension = 2, hardCut = True):
    def extendLine(line, extension): # (a.x,a.y,b.x,b.y)
        ab = line[2:4] - line[0:2]
        v = (ab / np.linalg.norm(ab)) * extension
        return [line[0:2] - v, line[2:4] + v]
    
    print(lineExtension)
    G = nx.Graph()
    for i, line1 in enumerate(lines):
        shLine1 = extendLine(line1, lineExtension)
        shLine1 = LineString(shLine1)
        for j, line2 in enumerate(lines[(i+1):]):
            shLine2 = extendLine(line2, lineExtension)
            shLine2 = LineString(shLine2)
            if shLine1.intersects(shLine2):
                # p = shLine1.intersection(shLine2)
                G.add_edge(i, i+j+1)
            elif not G.has_node(i):
                G.add_node(i)
            elif not G.has_node(i+j+1):
                G.add_node(i+j+1)
    out = np.array([]).reshape(0,4)
    comps = nx.algorithms.components.connected_components(G)
    if hardCut:
        comps = np.array(list(comps))
        sorted = np.array([len(x) for x in comps]).argsort()[::-1]
        comps = comps[sorted][:2]
    for comp in comps:
        if len(comp) >= min_components:
            indices = np.asarray(list(comp))
            out = np.concatenate((out, lines[indices]), axis=0)
    return out

def selectInOrderGenerator(size):
    out = [0,0]
    yield np.asarray(out)
    while out[0] != size - 2 or out[1] != size -1:
        out[1] += 1
        if out[1] == size:
            out[0] += 1
            out[1] = out[0]+1
        yield np.array(out)

def showImgWithLines(image, lines, title='img_with_lines', waitKey=True):
    img_with_lines = np.copy(image)
    for line in lines:
        line = line.astype(np.int32)
        img_with_lines = cv2.line(img_with_lines, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)
        img_with_lines = cv2.circle(img_with_lines, (line[0], line[1]), 2, (255, 80, 0), 3)
        img_with_lines = cv2.circle(img_with_lines, (line[2], line[3]), 2, (255, 80, 0), 3)
    cv2.imshow(title, img_with_lines)
    if waitKey:
        cv2.waitKey(0)

def computeLineScore(projectedLines, lines, angleTh = 4):
    score = 0
    for pLine in projectedLines:
        v1 = pLine[2:4] - pLine[0:2]
        len1 = np.linalg.norm(v1)
        minScore = 1e5
        mini = -1
        for i, line in enumerate(lines):
            localScore = 0
            v2 = line[2:4] - line[0:2]
            len2 = np.linalg.norm(v2)
            dot = abs(np.dot(v1 / len1, v2 / len2))
            dot = min(1, dot)
            angle = np.arccos(dot) * 180 / np.pi
            if angle < angleTh:
                dist1 = pointLineMinDist(pLine, line[0:2])
                dist2 = pointLineMinDist(pLine, line[2:4])
                dist3 = pointLineMinDist(line, pLine[0:2])
                dist4 = pointLineMinDist(line, pLine[2:4])
                minDist = min((dist1, dist2, dist3, dist4))
                if minDist < 50:
                    localScore = (angleTh-angle) * 200
                    dist1 = np.sum((pLine[0:2] - line[0:2])**2)
                    dist2 = np.sum((pLine[0:2] - line[2:4])**2)
                    dist3 = np.sum((pLine[2:4] - line[0:2])**2)
                    dist4 = np.sum((pLine[2:4] - line[2:4])**2)
                    localScore += (min(dist1, dist2) + min(dist3, dist4))**2
                    localScore += minDist**2
                    if minScore > localScore:
                        minScore = localScore
                        mini = i
        if mini != -1:
            lines = np.delete(lines, mini, axis=0)
        score += 1e3 - minScore
    return score

def test_single_image(inference, impath, output_path = "", threshold = 0.97):
    image = cv2.imread(impath)
    lines = inference.evaluate(image)
    nlines = len(lines)
    print('number of lines: ', nlines)

    ### VISUAL DEBUG ###
    # showImgWithLines(image, lines, 'nofilter', True)
    ### END VISUAL DEBUG ###

    # line based filtering
    lines = linesFiltering(lines, image.shape[:2])
    print('lines removed: ', nlines - len(lines), "\t remaining: ", len(lines))

    # graph base filtering (a line is connected with another if they intersect)
    nLines = len(lines)
    print(image.shape[:2])
    lines = linesFilteringWithGraph(lines, lineExtension=(min(image.shape[:2])/10))
    print('removed lines: ', nLines - len(lines), "\t remaining: ",len(lines))

    ### VISUAL DEBUG ###
    # showImgWithLines(image, lines, "filter", True)
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
    best_score = float('-inf')
    best_fitting_points = []
    best_projected_points = None

    ### VISUAL DEBUG ###
    # original_points = np.asarray([[134,361],[662,369],[651,235],[292,235]])
    # tennis_court_point_idx = np.asarray([4,5,7,6])
    
    # print(np.amax(tennis_court_model_points, axis=0).shape)
    # img_with_projected_lines = np.zeros(np.append(np.amax(tennis_court_model_points, axis=0)[[1,0]], [3]))
    # for idx in tennis_court_point_idx:
    #    img_with_projected_lines = cv2.circle(img_with_projected_lines, tennis_court_model_points[idx].astype(np.int32), 4, (255, 0, 0), thickness=-1)
    
    # cv2.imshow('model', img_with_projected_lines)
    # cv2.waitKey(0)
    
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
    
    # # print("best_rtmse: ", best_rtmse)
    # img_wrap = cv2.warpPerspective(image, np.linalg.inv(RT_matrix), (144, 312))
    # cv2.imshow('img_wrap', img_wrap)
    # cv2.waitKey(0)
    ### END VISUAL DEBUG ###

    model_image = np.zeros(np.amax(tennis_court_model_points, axis=0)[[1,0]] + 1)
    for line in tennis_court_model_lines:
        model_image = cv2.line(model_image, tennis_court_model_points[line[0]], tennis_court_model_points[line[1]], (255), thickness=2)

    lineGenerator = selectInOrderGenerator(lines.shape[0])
    modelLineGenerator = selectInOrderGenerator(tennis_court_model_lines.shape[0])

    for i in range(10000):
        # select_lines_idx = np.random.choice(lines.shape[0], size=(9,), replace=False)
        # select_points_idx = np.random.choice(points.shape[0], size=(4,), replace=False) # scegliere in base a linee casuali invece di punti casuali
        # select_model_lines_idx = np.random.choice(tennis_court_model_lines.shape[0], size=(6,), replace=False)
        # model_points_idx = np.random.choice(tennis_court_model_points.shape[0], size=(4,), replace=False)

        if  i == 0:
            select_model_lines_idx = next(modelLineGenerator)
        try:
            select_lines_idx = next(lineGenerator)
        except StopIteration:
            lineGenerator = selectInOrderGenerator(lines.shape[0])
            select_lines_idx = next(lineGenerator)
            try:
                select_model_lines_idx = next(modelLineGenerator)
            except StopIteration:
                break

        select_points = np.asarray([np.append(lines[select_lines_idx,0], lines[select_lines_idx,2]),np.append(lines[select_lines_idx,1], lines[select_lines_idx,3])]).T
        # print("select_points")
        # print(select_points)
        # select_points = points[select_points_idx]
        
        select_model_points = tennis_court_model_points[np.append(tennis_court_model_lines[select_model_lines_idx,0], tennis_court_model_lines[select_model_lines_idx,1])]
        # select_model_points = tennis_court_model_points[np.append(tennis_court_model_lines[:,0], tennis_court_model_lines[:,1])]
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
        if RT_matrix is None or np.sum(np.isinf(RT_matrix)) != 0:
            print("RT matrix contains an infinite")
            continue
        # if np.sum(mask) != 4:
        #     print("mask is different than 4")
        #     continue
        # img_wrap = cv2.warpPerspective(image, RT_matrix, (78, 36))
        # print(img_wrap.shape)
        # cv2.imshow('img_wrap', img_wrap)
        # cv2.waitKey(0)

        tennis_court_projected_points = RT_matrix @ np.r_[tennis_court_model_points.T, np.full((1, tennis_court_model_points.shape[0]), 1, dtype=np.float32)]
        if 0 in tennis_court_projected_points[2]:
            continue
        tennis_court_projected_points = tennis_court_projected_points / tennis_court_projected_points[2]
        tennis_court_projected_points = tennis_court_projected_points.T
        
        # print(tennis_court_projected_points)
        ### VISUAL DEBUG ###
        # img_with_projected_lines = np.copy(image)
        # for line in tennis_court_model_lines:
        #     img_with_projected_lines = cv2.line(img_with_projected_lines, tennis_court_projected_points[line[0]][0:2].astype(np.int32), tennis_court_projected_points[line[1]][0:2].astype(np.int32), (255, 0, 0), thickness=2)
        # for model_point in select_model_points:
        #     img_with_projected_lines = cv2.circle(img_with_projected_lines, model_point[0:2].astype(np.int32), 4, (255, 0, 0), thickness=-1)
        # for select_point in select_points:
        #     img_with_projected_lines = cv2.circle(img_with_projected_lines, select_point.astype(np.int32), 2, (0, 255, 0), thickness=-1)
        # cv2.imshow('img_with_projected_lines', img_with_projected_lines)
        # cv2.waitKey(0)
        ### END DEBUG ###

        projected_lines = []
        for line in tennis_court_model_lines:
            projected_lines.append(np.append(tennis_court_projected_points[line[0]][0:2], tennis_court_projected_points[line[1]][0:2]))
        projected_lines  = np.asarray(projected_lines)
        score = computeLineScore(projected_lines, lines)

        if abs(np.linalg.det(RT_matrix)) < sys.float_info.epsilon:
            print("Determinant equal to zero")
            continue

        ### VISUAL DEBUG ###
        # inverse_matrix = np.linalg.inv(RT_matrix)

        # tennis_court_reprojected_points = inverse_matrix @ tennis_court_projected_points.T
        # tennis_court_reprojected_points = tennis_court_reprojected_points / tennis_court_reprojected_points[2]
        # tennis_court_reprojected_points = tennis_court_reprojected_points.T

        # print("RT_matrix:")
        # print(RT_matrix)
        # img_with_projected_lines = np.zeros(np.amax(tennis_court_model_points, axis=0)[[1,0]])
        # for line in tennis_court_model_lines:
        #    img_with_projected_lines = cv2.line(img_with_projected_lines, tennis_court_reprojected_points[line[0]][0:2].astype(np.int32), tennis_court_reprojected_points[line[1]][0:2].astype(np.int32), (255), thickness=2)
        # cv2.imshow('model', img_with_projected_lines)
        # cv2.waitKey(0)
        ### END VISUAL DEBUG ###


        ### VISUAL DEBUG ###
        # img_wrap = cv2.warpPerspective(image, np.linalg.inv(RT_matrix), (model_image.shape[1], model_image.shape[0]))
        # img_wrap_gray = cv2.cvtColor(img_wrap, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('img_wrap_gray', img_wrap_gray)
        # cv2.waitKey(0)
        ### END VISUAL DEBUG ###

        if best_score < score:
            best_score = score
            best_RT_matrix = RT_matrix
            best_fitting_points = select_points
            best_projected_points = tennis_court_projected_points
        
        if i% 50 == 0:
            print("\rfitting attempts: ",i,"  best score: ", best_score, end='')
    
    best_fitting_points = np.asarray(best_fitting_points)

    print("best_score:", best_score)
    img_with_projected_lines = np.copy(image)
    for line in tennis_court_model_lines:
        img_with_projected_lines = cv2.line(img_with_projected_lines, best_projected_points[line[0]][0:2].astype(np.int32), best_projected_points[line[1]][0:2].astype(np.int32), (255, 0, 0), thickness=2)
    cv2.imwrite(output_path, img_with_projected_lines)
    # cv2.imshow('window', img_with_projected_lines)
    # cv2.waitKey(0)
    
    # print("best_rtmse: ", best_rtmse)
    # img_wrap = cv2.warpPerspective(image, np.linalg.inv(RT_matrix), (144, 312))
    # cv2.imshow('img_wrap', img_wrap)
    # cv2.waitKey(0)

def test(args):
    inference = LETRInference(args.checkpoint_filepath, lines_score = args.threshold_letr_score)

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
                test_single_image(inference, os.path.join(args.img_directory, impath), output_path = output_path, threshold = args.threshold)
    else:
        output_path = ""
        if args.output_path != "":
            output_path = args.output_path
        test_single_image(inference, os.path.join(args.img_directory, args.img), output_path = output_path, threshold = args.threshold)

if __name__ == "__main__":
    args = argument_parsing()

    test(args)