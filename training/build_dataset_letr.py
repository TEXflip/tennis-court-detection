import argparse
import sys
from typing import Tuple
from lxml import etree
import cv2
import os
import glob
import numpy as np
import json
import random

def argument_parsing():
    parser = argparse.ArgumentParser(description='LETR dataset creator')
    parser.add_argument("cvat_annotations_filepath",default="",type=str,
                        help="the annotations file")
    parser.add_argument("img_directory",default="",type=str,
                        help="input images directory")
    parser.add_argument("output_dirpath",type=str,
                        help="directory where store the results")
    parser.add_argument("--test_cvat_annotations_filepath",type=str,
                        help="the test annotations file", default="")
    parser.add_argument("--test_img_directory",type=str,
                        help="the input images directory", default="")
    args = parser.parse_args()
    if ('test_cvat_annotations_filepath' in args) != ('test_img_directory' in args):
        print("Both test parameter must be present or absent at the same time")
        sys.exit(1)
    if 'test_cvat_annotations_filepath' in args:
        if not os.path.isfile(args.test_cvat_annotations_filepath):
            print("Given test annotation file do not exist")
            sys.exit(1)
        if not os.path.isdir(args.test_img_directory):
            print("Given test image directory do not exist")
            sys.exit(1)
    if not os.path.isfile(args.cvat_annotations_filepath):
        print("Given annotation file do not exist")
        sys.exit(1)
    if not os.path.isdir(args.img_directory):
        print("Given image directory do not exist")
        sys.exit(1)
    return args

class DatasetCreator():
    def __init__(self, images_directory, cvat_annotations_filepath, output_dirpath, test_cvat_annotations_filepath="", test_img_dirpath=""):
        self.__images_directory = images_directory
        self.__tree_cvat_xml_root = etree.parse(cvat_annotations_filepath).getroot()
        self.__output_dirpath = output_dirpath

        if test_cvat_annotations_filepath != "":
            self.__test_tree_cvat_xml_root = etree.parse(test_cvat_annotations_filepath).getroot()
        self.__test_img_dirpath = test_img_dirpath

        self.__test_annotations = []
        self.__train_annotations = []

        self.__saved_images = set()

    
    def __parse_anno_file(self, image_name : str, test : bool = False) -> Tuple[list, bool]:
        anno = []
        image_name_attr = ".//image[@name='{}']".format(image_name.strip())
        found_on_annotation = False
        for image_tag in (self.__test_tree_cvat_xml_root if test else self.__tree_cvat_xml_root).iterfind(image_name_attr):
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

    def __process_image(self, img_filepath : str, test : bool = False) -> bool:
        print("Processing image ", img_filepath)
        image_filename = os.path.basename(img_filepath)
        frame_id = os.path.splitext(image_filename)[0]
        anno, found_on_annotation = self.__parse_anno_file(image_filename, test = test)
        if not found_on_annotation:
            print("Annotation not found for file: ", img_filepath)
            return False
        
        img = cv2.imread(img_filepath)

        output_frame_id = frame_id

        output_frame_id = random.randint(0, 9999999)
        while output_frame_id in self.__saved_images:
            output_frame_id = random.randint(0, 9999999)
        
        output_frame_id = "{:07}".format(output_frame_id)

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
        
        img_output_path = os.path.join(self.__output_dirpath, "images", output_frame_id + '.png')
        cv2.imwrite(img_output_path, img)
        
        dict_to_save = {
            'filename': output_frame_id + '.png',
            'height': img.shape[0],
            'width': img.shape[1],
            'lines': gt_lines_np.tolist(),
        }
        if test:
            self.__test_annotations.append(dict_to_save)
        else:
            self.__train_annotations.append(dict_to_save)
        
        return True
    
    def __end_processing(self):
        if self.__test_img_dirpath != "":
            with open(os.path.join(self.__output_dirpath, 'valid.json'), 'w') as file:
                json.dump(self.__test_annotations, file)
            test_filename = [annotation['filename'] for annotation in self.__test_annotations]
            with open(os.path.join(self.__output_dirpath, 'test.txt'), 'w') as file:
                file.writelines(test_filename)
        with open(os.path.join(self.__output_dirpath, 'train.json'), 'w') as file:
            json.dump(self.__train_annotations, file)
    
    def create_datasets(self):
        image_attr = ".//image"
        os.makedirs(os.path.join(self.__output_dirpath, "images"), exist_ok=True)
        image_files = glob.glob(os.path.join(self.__images_directory, '*.jpg'))
        image_files += glob.glob(os.path.join(self.__images_directory, '*.jpeg'))
        for image in image_files:
            self.__process_image(image)
        
        if self.__test_img_dirpath != "":
            test_image_files = glob.glob(os.path.join(self.__test_img_dirpath, '*.jpg'))
            test_image_files += glob.glob(os.path.join(self.__test_img_dirpath, '*.jpeg'))
            for image in test_image_files:
                self.__process_image(image, test=True)
        
        self.__end_processing()

def build_dataset(cvat_annotations_filepath : str, img_directory : str, output_path : str, test_cvat_annotations_filepath : str="", test_img_dirpath : str =""):
    dataset_creator = DatasetCreator(img_directory, cvat_annotations_filepath, output_path, test_cvat_annotations_filepath=test_cvat_annotations_filepath, test_img_dirpath=test_img_dirpath)

    dataset_creator.create_datasets()

def main():
    args = argument_parsing()

    build_dataset(args.cvat_annotations_filepath, args.img_directory, args.output_dirpath, test_cvat_annotations_filepath=args.test_cvat_annotations_filepath, test_img_dirpath=args.test_img_directory)

if __name__ == "__main__":
    main()