# Tennis court detection

This repository contain the code that given a image or a series of images, fit a tennis court model inside.

### Repository structure:

* ```modelFitting.py```: run with HAWP model and line scoring.
* ```modelFitting_letr.py```: run using the LETR model and line scoring.
* ```training``` contain all the scripts to build the datasets to transfer the neural networks
* ```testing-dataset```: dataset with sample images of tennis, basket and football courts, it is used to test the performance of the algorithm, it include also the annotation file for the tennis court fields
* ```training-dataset```: dataset with sample images of tennis courts from various viewpoint, contain also the annotation file used to make transfer learning to LETR

The other branch contain the same file structure, but use different system to score the line fitting. We suggest to use the main branch that are the one that have shown the best results.

## How run

### Installation

First clone the repository:
```bash
git clone https://github.com/TEXflip/sport-court-detection.git
git submodule init
git submodule update --remote
```

To run the system need _Python_ installed. Currently the system support _Python_ <= 3.8 (mainly for the HAWP part while the LETR part support also _Python_ 3.9). Most of the package listed below can be installed with recent versions of _pip_.

The required packages to install using are:
- ```scikit-learn``` tested with __0.24.2__
- ```torch``` tested with __1.7.1__ (but also more recent version are known to work). The program is set to automatically detect if a GPU is available and otherwise run it on CPU. 
- ```torchvision``` tested with __0.8.2__. Version older that this missing some required functions
- ```numpy``` tested with __1.19.5__. Other versions probably work without problems
- ```matplotlib``` tested with __3.3.4__. Other versions probably work without problems
- ```PIL``` tested with __8.1.0__. Other versions probably work without problems
- ```CocoAPI```. We suggested to install it directly from source with the command ```pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'```
- ```docopt``` tested with __0.6.2__. Other versions probably work without problems
- ```cv2``` tested with __4.5.1__. We usually recommend to install OpenCV from source. Other version probably work as well, but must be made special attention to the relation with ```PyTorch```


### Pretrained models

The already pretrained model is available at [https://github.com/TEXflip/sport-court-detection/raw/main/pretrained-model/letr_best_checkpoint.pth](https://github.com/TEXflip/sport-court-detection/raw/main/pretrained-model/letr_best_checkpoint.pth).

### Building dataset for LETR transfer learning

To create the dataset run the following command, replacing the part within square bracket with the file locations.
```bash
python3 training/build_dataset_letr.py [training annotations filepath] [training image directory] [output directroy dirpath] --test_cvat_annotations_filepath [filepath to the testing annotation]
--test_img_directory [dirpath to the directory of the image to use as test]
```

### LETR transfer learning

LETR must be trained in segments. While it is available as pretrained the entire model, the single pieces are not available, and it is not possible to start the transfer learning of first segment from the final model. So it is first necessary to download the Wireframe Dataset, please check the [LETR repository](https://github.com/mlpc-ucsd/LETR) on how to do it.
The run this command to train the first part, replace the square bracket parts with the correct path:

```bash
PYTHONPATH=$PYTHONPATH:./LETR/src python -m torch.distributed.launch \
    --master_port=$((1000 + RANDOM % 9999)) --nproc_per_node=8 --use_env  src/main.py --coco_path [wireframe processed dirpath] \
    --output_dir [output dirpath] --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --batch_size 1 --epochs 500 --lr_drop 200 --num_queries 1000  --num_gpus 8   --layer1_num 3 | tee -a [output dirpath]/history.txt
```

Then for start transfer learning run the following command, replacing the dirpath as before:
```bash
PYTHONPATH=$PYTHONPATH:./LETR/src python -m torch.distributed.launch \
    --master_port=$((1000 + RANDOM % 9999)) --nproc_per_node=8 --use_env src/main.py --coco_path [tennis court dataset dirpath] \
    --output_dir [output dirpath] --backbone resnet50 --resume [Last output dirpath] \
    --batch_size 1 --epochs 1000 --lr_drop 200 --num_queries 1000  --num_gpus 8   --layer1_num 3 | tee -a [output dirpath]/history.txt
```

You can then train the other layers:
```bash
PYTHONPATH=$PYTHONPATH:./LETR/src python -m torch.distributed.launch \
    --master_port=$((1000 + RANDOM % 9999)) --nproc_per_node=8 --use_env  src/main.py --coco_path [tennis court dataset dirpath] \
    --output_dir [output dirpath] --LETRpost --backbone resnet50 --layer1_frozen --frozen_weights [stage 1 TL checkpoint] --no_opt \
    --batch_size 1 ${@:2} --epochs 300 --lr_drop 120 --num_queries 1000 --num_gpus 8 | tee -a [output dirpath]/history.txt  
PYTHONPATH=$PYTHONPATH:./LETR/src python -m torch.distributed.launch \
        --master_port=$((1000 + RANDOM % 9999)) --nproc_per_node=8 --use_env  src/main.py --coco_path [tennis court dataset dirpath] \
        --output_dir [output dirpath]  --LETRpost  --backbone resnet50  --layer1_frozen  --resume [stage 2 TL checkpoint]  \
        --no_opt --batch_size 1  --epochs 25  --lr_drop 25  --num_queries 1000  --num_gpus 8  --lr 1e-5  --label_loss_func focal_loss \
        --label_loss_params '{"gamma":2.0}'  --save_freq 1  |  tee -a [output dirpath]/history.txt 
```

At this point it is possible to use the checkpoint of the last stage to evaluate the performance of the system.

### Run the LETR-based system

For running the LETR-based system with line scoring use the following command, replacing the square brackets with the corresponding dirpath.

```bash
PYTHONPATH=$PYTHONPATH:./LETR/src python modelFitting_letr.py --checkpoint-filepath [last stage checkpoint filepath] --img [image filepath] --output_path [dirpath where save the result]
```

### Run the HAWP-based system

To run the HAWP-based pipeline is it necessary to first download the pretrained model, please reference to the [HAWP repository](https://github.com/cherubicXN/hawp) for it.

Then it is possible to run the pipeline using the command (replace the square brackets with the corresponding paths):

```bash
python modelFitting.py --config-file [hawp config filepath, in the default config should be hawp/config-files/hawp.yaml] --img [image filepath] --output_path [dirpath where save the result]
```


## How it works:

* The image is feeded to one of the Line Detection Neural Networks
* The output is a set of lines defined as a couple of 2 points
* Then the lines are filtered using 3 different filters
* The resulting lines are used to find an homography
* for each pair of lines of both lines output and model template, 4 points are used to find the homography and then the projection (using the resulting matrix) is evaluated using one of the two scoring techniques
* The projection with the best scoring is the final result

In the image below is illustrated the flow of the image

![](./assets/scheme.png)

### Filters:

In order to reduce the number of lines, 3 filters have been implemented:
![](./assets/filters.png)

#### Line Filter:

It mainly removes the overlapped lines.

for every couple of lines AB and CD: if the angle of the intersection between AB and CD is smaller than a threshold and min(AC,AD,BC,BD) < threshold keeps the shorter line.


#### Mask Filter:

It removes the lines not overlapping the white (or the color of the court lines) pixels.

* create a mask *LinesMask* from n×m black image and draw the lines (with thickness=6px)
* apply mask on the image
* Init a gaussian mixture with 3 gaussians and the masked image
* get the gaussian g fitting (255,255,255) color or the court line color
* produce a mask *CandidateLinesMask* by selecting the pixels fitted with g and applying *LinesMask*
* for each line:
    - produce a n×m$ black image and draw the line
    - get the number of pixels p overlapped with *CandidateLinesMask*
    - keep the lines with p>0.5*(length of the line)

#### Graph Filter:

It removes lonely lines and it keeps only big intersected groups of lines.

* extend the lines with min(n,m)/20
* init graph G with set of nodes = set of lines
* for each couple of line a,b:
    - if a intersect b, connect them on the graph G
* compute the connected components of G
* keep only the components > 3 (or in hard mode keep only biggest 2 connected components)

### Homography

The homography is computed with ```findHomography()``` function of openCV (so RANSAC or other algorithms can be used).

Then is searched the best homography trying all the combinations of the points of 2 lines with 2 lines of the model template and using a scoring.

### Scoring

In this project have are been implemented 3 scoring techniques: template matching based, gaussian mixture based and line based, but only the last 2 give reasonable results and will be described.

#### Gaussian-based Scoring

* create a mask *LinesMask* from n×m black image and draw the lines (with thickness=6px)
* apply *LinesMask* on the image
* Init a gaussian mixture with 3 gaussians and the precedently masked image
* get the gaussian *g* fitting (255,255,255) color or the court line color
* get the projected lines of the model template using the Homography matrix
* create a mask *ModelLinesMask* using the projected lines
* apply ModelLinesMask on the image
* the score is computed as the sum of the pixels of the precedently masked image predicted with *g*

#### Line-based Scoring

![line-scoring](./assets/line-scoring.PNG)

* for each pair of lines compute the local score only if distance(AB, CD) < *distance threshold* and α < *angle threshold*
* local score(AB, CD)=min(AC,BD,AD,BC)² + (min(AC,CB) + min(BD,BC))² - 200α
* total score is the sum of the smallest local score of each pair of lines (so smaller is better)

## Results

### HAWP with Gaussian Scoring

![HAWP_gaussian_scoring](./assets/HAWP_gaussian_scoring.PNG)

### HAWP with Line Scoring

![HAWP_line_scoring](./assets/HAWP_line_scoring.PNG)

### LETR with Gaussian Scoring

![LETR_gaussian_scoring](./assets/LETR_gaussian_scoring.png)

### LETR with Line Scoring

![LETR_line_scoring](./assets/LETR_line_scoring.png)
