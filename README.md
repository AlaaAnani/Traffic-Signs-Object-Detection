# Traffic-Signs-Object-Detection
This is an assignment for the CSCE460301 - Fundamental of Computer Vision (2021 Spring) course at AUC. All copy rights © go to Alaa Anani. Course Link: http://catalog.aucegypt.edu/preview_course_nopop.php?catoid=36&coid=83731

For the full hierarchy of files (include the code and the dataset of images), download the full directory found here: https://drive.google.com/drive/folders/1-cqcnYu42tIOig9Wq3ndwU8rdNP7Fcpe?usp=sharing

# Steps for Red-and-Yellow Road Signs Detection

## 1. Red Color Segmentation

An important thing I noticed is that all road signs of interest are bounded by red, making red detection more important than yellow. I first segment the color red in the image by detecting pixels lying in a certain red range I defined using the `cv2.inRange` function. Then I zero all other non-red pixels to only leave the red pixels being non-zero in the resulting segmented image `red_segmented_bgr`. 

## 2. Binarizing Red-Segmented Images

Then I use `cv2.threshold` function to binarize the image, which returns a new gray image of pixels with binary values where the red pixels in `red_segmented_bgr` are set to 1 and the rest are 0. The resulting image in called `binary_img`.

## 3. Getting Proposed Regions

Using the function `cv2.findContours`, I get all the bounding boxes in the simple resulting `binary_img` from the previous step. Few of these rectangles will definitely contain our desired objects (red+yellow road signs) since they bound any red-containing region in the image. This method is much more efficient than another method I tried earlier, which was Selective Search. The proposed number of regions proposed decreased from around 7k+ in Selective Search to few tens or hundreds in this method, depending on how many red regions are found in the image.

## 4. Running a Binary CNN-Classifier on Proposed Regions

To know which of the proposed regions exactly contains the desired road signs, I run a pre-trained classifier on all of thenm first. Then, I order them descensingly according to their probabilities outputted from the classifier. Then, I take the n-highest regions passing a certain threshold (set as `0.9` by default, could be changed) to be containing the detected object. Green rectangles show a correctly detected object by the classifier, while blue ones show all proposed regions in general.

# Training a Binary Classifier

**Dataset**: I made a dataset of 2 classes composed of 513 images by taking cropped screenshots from the given set of sign images. The `1` class contains all given signs of interest and scraped red-and-yellow signs from the internet to make the set more diverse. The `0` class is constructed through running Selective Search and also getting the cropped regions suggested by step `3` and humanly specifiying which of them belongs to class `0`. Selective Search provided much diversity since it proposes many regions, not only red-containing regions. Regions from step `3` served as a very strong basis for the model to be robust again red regions that are not red-and-yellow road signs. The dataset is the folder `dataset/`.

**Model Architecture:** found in `cnn_model.py` and the best model is saved as `cnn_model_2`. 
**Loss:** Binary cross entropy loss function is used.

# Results

`Number of correct detections = 34` with regions covering nearly 100% of the object

`Number of false detections = 0`.


The following code cells show a demo of the full assignment requirements. 

## Imports and Setup
```
import argparse
import random
import time
import cv2
from skimage.io import imread, imshow, imsave
import numpy as np
import os
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow import keras
from object_detection import segment_red, binarize_segments, get_proposed_rectangles, detect_road_signs
from utils import show
base = "Images/"
signs_model = keras.models.load_model('cnn_model_2')
titles = ['Original Image', 'Binary Red-Segmented Image', 'All Proposed Rectangles', 'Detected Road Signs = ']
images_paths = [base+entry for entry in os.listdir(base)]
```

## Code Logic 
```
for i, path in enumerate(images_paths):
    image = cv2.imread(path)
    red_segmented_bgr = segment_red(image)
    binary_img = binarize_segments(red_segmented_bgr)
    rects = get_proposed_rectangles(binary_img)
    output_w_all_rects, final_output, detected_signs, _ = detect_road_signs(signs_model, image, rects)
    images = [image, binary_img, output_w_all_rects, final_output]
    titles[2] = 'Overall Proposed Bounding Rectangles = ' + str(len(rects))
    titles[3] = 'Detected Road Signs = ' + str(detected_signs)
    show(2, 2, images, titles, save=False)
```

# Output Plots
<kbd>![image](https://drive.google.com/uc?export=view&id=1-SUrWSLbzfLFF31JFyqed71h_Q1JN534)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-TpajfOjrrlNTM6yDmDLtoLyjfTTv7Qb)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-TVwNnZq8Gj2XCXjvHEw878myzGo3okW)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-TtxMDM4tNPEIjYnfI8PmrPH_boKcNEn)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-RtGEBPtmgNefO99m3s3NRzwetuI4sai)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-URLyAY1oZ5Ds4LnmiCnTv4cr_srMKpW)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-UbVM-kseuwyBQOgyrpsBX0P1soZrMHd)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-V7GWlP55k1O0fg7EPz5vLBlCYO_hAaL)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-WeswzGRlQsKY5zAMl4cJYafbdalPbDE)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-WQLUQU694eUmYWUUQ5KniKdB7WHC37_)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-YWS5SJlQ0dQQYfKtIppze5r5PWmmcNV)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-XJjcY8iZIdpepnnchuzQrKplHwuFI1E)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-YbageFCd5j0K3A1WAjaexSCFGRPGovA)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-YpXWns7Y_cZ7CtBa1deKIZyPyQ_-WhX)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-Yh5pSqwDvyOwjTWGPucOgXbQPD9ySSt)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-Yw5KQt5bgs2Qp_z9ijA35J2WJ51XpmQ)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-ZLjzdIvCAfntfeJx5SJ6jTIjXG8qTX7)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-Z-VmhMGcFK37HECP1CJMFxyu7P-EpQA)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-ZpmzhzyWItCiymMJl4s-ykL4ko5ceQk)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-ZTAo1SzVo30i3XUxHmJN9xJ5q8iADk1)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-_UgMMnbbVuUU9Oed1UeCxvjFqwkC_zm)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-_6EONmuLjcwiB_Tmtvb929WwK5lJIPe)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-aabTEMb2WXOqbjYuqw58lQUvSZBCzGr)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-bDxMlJn8K6ZMFa5TwX-feAbSKfvsbvP)</kbd>
#
<kbd>![image](https://drive.google.com/uc?export=view&id=1-beIEH8OOqMhFjoCP2d6DfRnB6vV1v77)</kbd>
