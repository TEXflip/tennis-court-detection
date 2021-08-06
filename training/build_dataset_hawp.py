"""
FOR REFERENCE ONLY
Training tennis court line detection using HAWP do not work because require negative and positive edge (not existing in court detection)
"""


import argparse
import sys
from lxml import etree
import cv2
import os
import glob
import numpy as np
import json

def argument_parsing():
    parser = argparse.ArgumentParser(description='Efficient Homographic dataset creator')
    parser.add_argument("cvat_annotations_filepath",default="",type=str,
                        help="the annotations file")
    parser.add_argument("img_directory",default="",type=str,
                        help="input images directory")
    parser.add_argument("output_dirpath",type=str,
                        help="directory where store the results")
    parser.add_argument("--test",type=bool,
                        help="save the result under the form of test file")
    args = parser.parse_args()
    if not os.path.isfile(args.cvat_annotations_filepath):
        print("Given annotation file do not exist")
        sys.exit(1)
    return args

class DatasetCreator():
    def __init__(self, images_directory, cvat_annotations_filepath, output_bev_dirpath, output_rotation_matrix_dirpath, output_original_dirpath, output_lines_dirpath, test):
        self.__images_directory = images_directory
        self.__tree_cvat_xml_root = etree.parse(cvat_annotations_filepath).getroot()
        self.__output_bev_dirpath = output_bev_dirpath
        self.__output_rotation_matrix_dirpath = output_rotation_matrix_dirpath
        self.__output_original_dirpath = output_original_dirpath
        self.__output_lines_dirpath = output_lines_dirpath

        self.__output_lines_annotations = []
        self.__test_mode = test

    
    def __parse_anno_file(self, image_name):
        anno = []
        image_name_attr = ".//image[@name='{}']".format(image_name)
        found_on_annotation = False
        for image_tag in self.__tree_cvat_xml_root.iterfind(image_name_attr):
            found_on_annotation = True
            image = {}
            for key, value in image_tag.items():
                image[key] = value
            image['lines'] = []
            for polyline_tag in image_tag.iter('polyline'):
                polyline = {'type': 'polyline'}
                for key, value in polyline_tag.items():
                    polyline[key] = value
                image['lines'].append(polyline)
            image['lines'].sort(key=lambda x: int(x.get('z_order', 0)))
            anno.append(image)
        return anno, found_on_annotation

    def __process_image(self, img_filepath):
        print("Processing image ", img_filepath)
        image_filename = os.path.basename(img_filepath)
        frame_id = os.path.splitext(image_filename)[0]
        anno, found_on_annotation = self.__parse_anno_file(image_filename)
        if not found_on_annotation:
            print("Annotation not found for file: ", img_filepath)
            return False
        
        img = cv2.imread(img_filepath)

        output_frame_id = ("test_" if self.__test_mode else "") + frame_id

        lines = []
        for record in anno:
            lines.extend(record['lines'])
        
        lines_type = {}
        for line in lines:
            if len(line['points']) == 0:
                print("Exist a line without any point")
                continue
            split_line = line['points'].split(';')
            if len(split_line) < 2:
                print("Not a valid line")
                continue
            points = [tuple(map(float, p.split(','))) for p in split_line[0:2]]
            if line['label'] in lines_type:
                lines_type[line['label']].append(points)
            else:
                lines_type[line['label']] = [points]
        
        if 'baseline' not in lines_type:
            print("Not found any baseline line")
            return False
        if 'sideline-doubles' not in lines_type:
            print("Not found any sideline for doubles")
            return False
        if 'sideline-singles' not in lines_type:
            print("Not found any sideline for singles")
            return False
        if 'service-line' not in lines_type:
            print("Not found any service line")
            return False
        if 'service-centerline' not in lines_type:
            print("Not found any service centerline")
            return False
        
        if len(lines_type['baseline']) != 2:
            print("Baseline lines are different than 2, found ", len(lines_type['baseline']))
            return False
        if len(lines_type['sideline-doubles']) != 2:
            print("Doubles sidelines are different than 2, found ", len(lines_type['sideline-doubles']))
            return False
        if len(lines_type['sideline-singles']) != 2:
            print("Singles sidelines are different than 2, found ", len(lines_type['sideline-singles']))
            return False
        if len(lines_type['service-line']) != 2:
            print("Service lines are different than 2, found ", len(lines_type['service-line']))
            return False
        if len(lines_type['service-centerline']) != 1:
            print("Service centerlines are different than 1, found ", len(lines_type['service-centerline']))
            return False

        gt_lines = []
        gt_lines.extend(lines_type['baseline'])
        gt_lines.extend(lines_type['sideline-doubles'])
        gt_lines.extend(lines_type['sideline-singles'])
        gt_lines.extend(lines_type['service-line'])
        gt_lines.extend(lines_type['service-centerline'])
        gt_lines_np = np.asarray(gt_lines)
        gt_lines_np = np.reshape(gt_lines_np, (gt_lines_np.shape[0], gt_lines_np.shape[1] * gt_lines_np.shape[2]))
        junc = np.concatenate((gt_lines_np[:, 0:2], gt_lines_np[:, 2:4]), axis=0)
        positive_edges = np.reshape(np.arange(0, gt_lines_np.shape[0] * 2), (2, gt_lines_np.shape[0])).T
        
        img_path_output_lines = os.path.join(self.__output_lines_dirpath, output_frame_id + '.png')
        cv2.imwrite(img_path_output_lines, img)
        
        dict_to_save = {
            'filename': output_frame_id + '.png',
            'height': img.shape[0],
            'width': img.shape[1],
            'junctions': junc.tolist(),
        }
        if self.__test_mode:
            dict_to_save['lines'] = gt_lines_np.tolist()
        else:
            dict_to_save['edges_positive'] = positive_edges.tolist()
            dict_to_save['edges_negative'] = []
        self.__output_lines_annotations.append(dict_to_save)
    
    def __end_processing(self):
        if self.__test_mode:
            name = 'test.json'
        else:
            name = 'train.json'
        with open(os.path.join(self.__output_lines_dirpath, name), 'w') as file:
            json.dump(self.__output_lines_annotations, file)
            self.__output_lines_annotations
    
    def create_datasets(self):
        image_files = glob.glob(os.path.join(self.__images_directory, '*.jpg'))
        image_files += glob.glob(os.path.join(self.__images_directory, '*.jpeg'))
        for image in image_files:
            self.__process_image(image)
        self.__end_processing()

def build_dataset(cvat_annotations_filepath, img_directory, output_path, test):
    output_bev_dirpath = os.path.join(output_path, "bev_views")
    os.makedirs(output_bev_dirpath, exist_ok=True)
    output_rotation_matrix_dirpath = os.path.join(output_path, "rotation_matrix")
    os.makedirs(output_rotation_matrix_dirpath, exist_ok=True)
    output_original_dirpath = os.path.join(output_path, "original_views")
    os.makedirs(output_original_dirpath, exist_ok=True)
    output_lines_dirpath = os.path.join(output_path, "lines")
    os.makedirs(output_lines_dirpath, exist_ok=True)

    dataset_creator = DatasetCreator(img_directory, cvat_annotations_filepath, output_bev_dirpath, output_rotation_matrix_dirpath, output_original_dirpath, output_lines_dirpath, test)

    dataset_creator.create_datasets()

    
    

def main():
    args = argument_parsing()

    build_dataset(args.cvat_annotations_filepath, args.img_directory, args.output_dirpath, args.test)

if __name__ == "__main__":
    main()