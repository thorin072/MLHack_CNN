from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
import time
from os import listdir
from xml.etree import ElementTree
from keras.preprocessing import image
# example of loading an image with the Keras API
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

# Класс описания конфигурации сетки
class CalculoryConfig(Config):
    # Конфигурационное имя
    NAME = "Calculory_config"
 
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # все классы продуктов + бэграунд
    NUM_CLASSES = 30+1
   
    # Кол-во шагов за эпоху
    STEPS_PER_EPOCH = 50
    
    # Скорость обучения
    LEARNING_RATE=0.006
    
    # Вероятность обноружения
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES=10

class CalculoryDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self,dataset_dir, is_train=True):
        
        # Все классы, которые будут использованы в обучении
        self.add_class("dataset", 1, "rise")
        self.add_class("dataset", 2, "eels on rice")
        self.add_class("dataset", 3, "pilaf")
        self.add_class("dataset", 4, "chicken-'n'-egg on rice")
        self.add_class("dataset", 5, "pork cutlet on rice")
        self.add_class("dataset", 6, "beef curry")
        self.add_class("dataset", 7, "sushi")
        self.add_class("dataset", 8, "chicken rice")
        self.add_class("dataset", 9, "fried rice")
        self.add_class("dataset", 10, "tempura bowl")
        self.add_class("dataset", 11, "bibimbap")
        self.add_class("dataset", 12, "toast")
        self.add_class("dataset", 13, "croissant")
        self.add_class("dataset", 14, "roll bread")
        self.add_class("dataset", 15, "raisin bread")
        self.add_class("dataset", 16, "chip butty")
        self.add_class("dataset", 17, "hamburger")
        self.add_class("dataset", 18, "pizza")
        self.add_class("dataset", 19, "sandwiches")
        self.add_class("dataset", 20, "udon noodle")
        self.add_class("dataset", 21, "tempura udon")
        self.add_class("dataset", 22, "soba noodle")
        self.add_class("dataset", 23, "ramen noodle")
        self.add_class("dataset", 24, "beef noodle")

        self.add_class("dataset", 95, "pizza toast")
        self.add_class("dataset", 96, "dipping noodles")
        self.add_class("dataset", 97, "hot dog")
        self.add_class("dataset", 98, "french fries")
        self.add_class("dataset", 99, "mixed rice")
        self.add_class("dataset", 100, "goya chanpuru")

          
        for i in range(1,24):
            # define data locations for images and annotations
            images_dir = dataset_dir +'\\'+str(i)+'\\'
            annotations_dir = dataset_dir +'\\'+str(i)+'\\xml'+'\\'
        
            # Iterate through all files in the folder to 
            #add class, images and annotaions
            for filename in listdir(images_dir):
          
                # extract image id
                image_id = filename[:-4]
            
                # setting image file
                img_path = images_dir + filename
            
                # setting annotations file
                ann_path = annotations_dir + image_id + '.xml'
                if i==1:
                    name_class='rise'
                if i==2:
                    name_class='eels on rice'

            
                
                # adding images and annotations to dataset
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path,class_ids=name_class)
        for i in range(95,100):
            # define data locations for images and annotations
            images_dir = dataset_dir +'\\'+str(i)+'\\'
            annotations_dir = dataset_dir +'\\'+str(i)+'\\xml'+'\\'
        
            # Iterate through all files in the folder to 
            #add class, images and annotaions
            for filename in listdir(images_dir):
          
                # extract image id
                image_id = filename[:-4]
            
                # setting image file
                img_path = images_dir + filename
            
                # setting annotations file
                ann_path = annotations_dir + image_id + '.xml'
                if i==95:
                    name_class='pizza toast'
                if i==96:
                    name_class='dipping noodles'
            
                # adding images and annotations to dataset
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path,class_ids=name_class)

# extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height
# load the masks for an image
    """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
     """
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        
        # define anntation  file location
        path = info['annotation']
        name_class=info['class_ids']
        # load XML
        boxes, w, h = self.extract_boxes(path)
       
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(name_class))
        return masks, asarray(class_ids, dtype='int32')
# load an image reference"""Return the path of the image."""
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']


config = CalculoryConfig()
config.display()

train_set = CalculoryDataset()
train_set.load_dataset(os.path.abspath(os.curdir)+'\\DATAEAT', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare test/val set
test_set = CalculoryDataset()
test_set.load_dataset(os.path.abspath(os.curdir)+'\\DATAEAT', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

image_ids = np.random.choice(train_set.image_ids, 1)
for image_id in image_ids:
    image = train_set.load_image(image_id)
    mask, class_ids = train_set.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, train_set.class_names)



#print("Loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')

#load the weights for COCO
model.load_weights(os.path.abspath(os.curdir)+'//mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

## train heads with higher lr to speedup the learning
model.train(train_set, test_set, learning_rate=2*config.LEARNING_RATE, epochs=2, layers='heads')
history = model.keras_model.history.history


model_path = os.path.abspath(os.curdir)+'\\mask_rcnn_'+ '.' + str(time.time()) + '.h5'
model.keras_model.save_weights(model_path)

print('DONE!')




