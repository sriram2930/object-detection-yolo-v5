# Object Detection Yolo-v5

Deep Neural Networks (DNNs) have emerged as a powerful tool for Object detection tasks in computer vision. Object detection focuses on locating the presence of object with a bounding box and detecting the class of located objects in these boxes. This project explores the application of deep learning techniques particularly convolution neural networks (CNNs), for the task of object detection. By using state-of-the-art architectures such as YOLO (You Only Look Once), we focus on implementing YOLO model for object detection tasks. Our project evaluates its effectiveness and accuracy in identifying and localizing objects in real-time scenarios. We explore techniques for model optimization and deployment. Additionally, we extend on seamless integration of YOLO model with webcam for live video streaming, providing a straightforward solution for real-time object detection directly from live video feeds. Our project aims to provide a practical framework for deploying object detection models in real-world applications.

# Introduction 

Object detection, which is a subfield of computer vision, has been widely applied in the domains such as autonomous driving, robotics, security surveillance, and healthcare. Robust real-time object detection and classification capabilities have emerged as extremely important in applications where quick, accurate, and dependable judgments must be made. However, real time object detection is still a computation intensive task which, normally needs hardware such as GPUs or TPUs. This poses a problem when it comes to implementation of the object detection models on resource constrained and power limited devices, especially in edge use cases that require portability and self-containment.
      ESP32 Cam module is an ideal module as it has the needed camera, and it has in-built capability for wireless communication all integrated at an affordable price.  
Nevertheless, if flexible approaches like DNNs are to be applied in this environment, they need to be optimized for it because of the platform’s little processing power and memory. However, conventional object detection frameworks such as YOLO, SSD and Faster R-CNN do not inherently lend themselves to such platforms hence requiring optimization and model quantization.
This research is centered around a deep neural network object detection system utilizing the ESP32 Cam module to understand how technologies with complex algorithms can be made to work on limited hardware. To achieve the goal of real-time object detection on ESP32, this work therefore proposes to use lightweight versions of the existing object detection models and apply optimization techniques including pruning and quantization of the models.

#	Dataset description
There are two primary datasets we used to train the YOLO model: the COCO dataset and the PASCAL VOC dataset.
i.	COCO (Common Objects in Context) Dataset
The COCO dataset consists of training, validation, and test images, along with the corresponding labels and target class annotations. It includes over 330,000 images and 80 object categories.
The bounding box annotations of this dataset has four parameters:
a.	(x, y): The coordinates of the top-left corner of the bounding box.
b.	Width (w): The horizontal length of the bounding box.
c.	Height (h): The vertical length of the bounding box.
COCO dataset also provides some metadata for each bounding box
a.	Category ID: Identifying the object class from the 80 available categories.
b.	IsCrowd: A flag indicating whether the bounding box covers a group of objects.
c.	Area: The area covered by the bounding box, calculated as w * h.
ii.	PASCAL VOC Dataset
The PASCAL VOC 2012 version includes 11,530 images and 20 object categories.
The bounding box annotations of this dataset are defined as:
a.	xmin, ymin: The coordinates of the top-left corner of the bounding box.
b.	xmax, ymax: The coordinates of the bottom-right corner of the bounding box.
The metadata includes:
a.	Class Label: The object class from 20 predefined categories.
b.	Pose: The viewpoint of the object.
c.	Truncated: A flag indicating whether the object extends outside the image boundary, i.e., if the bounding box is truncated.
d.	Occluded: A flag indicating whether the object is partially occluded by another object.

# Components Used

1.	ESP32-CAM Module
   <img width="262" height="174" alt="image" src="https://github.com/user-attachments/assets/d5bad57a-6878-4888-a84c-958647390070" />

  In this project, we utilized the ESP32-CAM to capture video streams, which were then processed for real-time inference using the YOLO model.The ESP32-CAM is a microcontroller that integrates video capturing, making it both affordable and user friendly. This compact module features an OV2640 camera and a microSD card slot, enabling easy data storage and retrieval. Additionally, it includes an onboard LED for flash and multiple GPIOs for peripheral connectivity, making it ideal for many embedded applications.
  
2.	FTDI Programmer
   
<img width="209" height="139" alt="image" src="https://github.com/user-attachments/assets/26a650c0-9594-41a1-bdb0-4c37d23c5574" />

We utilized an FTDI module in our project since it is crucial for uploading code to the ESP32-CAM, which does not feature a USB port for direct connection to a computer. The FTDI programmer functions as a USB-to-Serial adapter, converting USB signals from the computer into serial signals that the microcontroller can understand.

# Model Building Workflow
<img width="488" height="470" alt="image" src="https://github.com/user-attachments/assets/69b3dd5a-26ef-4616-9611-ce39826fd167" />

# Architecture
# YOLO v4
<img width="518" height="232" alt="image" src="https://github.com/user-attachments/assets/614a0c1c-59b2-4dd9-b3c1-66de2eeb4261" />

Yolo has overall 24 convolutional layers, four max-pooling layers, and two fully connected layers. The YOLO architecture operates as follows: the dimensions is first resized to have a fixed size of 448 by 448 pixels and then fed into convolutional neural network (CNN). This resizing is done to ensure that images that come in different sizes are processed uniformly which is key to high speed in the detection. The first layers of CNNs enable the number of filters to reduce the depth of the feature maps to avoid large parameters and computations. After that they use 3×3 convolution layer in order to get spatial information and produce a cuboidal output which contains feature maps of multiple scales.
Most layers of the network involve the ReLU (Rectified Linear Unit) activation function to bring non-linearity to the network to be able to capture complex patterns in the data. But in the last layer of the network, they have used linear activation function because it requires output exact values of the coordinates of the bounding box and confidence score of the detected object.
To enhance this performance and prevent overfitting YOLO has incorporated several techniques as follows. This preprocess is performed in all layers to normalize the inputs to each layer for faster learning and stable convergence. Moreover, regularization is done through dropout, this is done through abstaining random neurons during the training process to discourage dependency on these neurons further improving the feature of generalization

# YOLOv5 
YOLOv5 is an advancement over YOLOv4. One key feature is the AutoAnchor algorithm, which automatically adjusts anchor boxes during training to fit the dataset and image sizes better. YOLOv5 has a modified CSPDarknet53 backbone and introduces the SPPF (Spatial Pyramid Pooling Fast) layer, which speeds up the process of detecting objects. YOLOv5 comes in different sizes from YOLOv5n (nano) to YOLOv5x (extra-large) designed for various hardware setups. It also uses data augmentations like Mosaic and MixUp to improve training performance and stability.

 <img width="548" height="373" alt="image" src="https://github.com/user-attachments/assets/c4c92096-5986-4a4e-8211-7097d2886c51" />

# Results

The confusion matrix for each model provides a visual representation of true versus predicted classifications. It highlights the number of correct and incorrect predictions across various classes.
 The diagonal blocks are significantly darker, indicating a higher number of correct predictions. Only a few off-diagonal blocks are colored lightly, representing misclassifications, which are crucial for identifying areas for improvement.
Precision indicates proportion of true positive predictions among all positive predictions made by the YOLO model. 
      Precision (P) =TP /(TP+FP)	
A high recall score means the YOLO model is effective at identifying most of the actual objects in the images from the COCO dataset.
              Recall (R)= TP / (TP + FN)
The F1 score indicates a good balance between precision and recall. A high F1 score indicates that the model is not only identifying most objects but also doing so accurately.
F1-score = 2 (P x R) / (P + R)
The above metrics were mainly used to analyze various YOLO versions. Additionally, the average precision (AP) for each model was calculated at different Intersection over Union (IoU) thresholds, along with precision-recall (PR) curves for each model, illustrating the trade-offs between precision and recall. Performance metrics for various sizes of the YOLOv5 model evaluated on the COCO dataset is recorded below.
<img width="518" height="168" alt="image" src="https://github.com/user-attachments/assets/bab68c76-ae8c-48dd-8e91-8b2abb8ef40c" />

Confusion Matrix for YOLO Models on the COCO Dataset:

<img width="386" height="307" alt="image" src="https://github.com/user-attachments/assets/dd0ed118-8b6f-463b-a614-c5501834726f" />

Precision-Recall Curve of YOLOv5x:

<img width="418" height="278" alt="image" src="https://github.com/user-attachments/assets/c7af39d3-03d2-442f-b14a-f54a07b92b40" />

# Streamlit-based website displaying live object detection results 

<img width="468" height="300" alt="image" src="https://github.com/user-attachments/assets/9952b7db-7199-44d4-b627-2f266fa6aeed" />

<img width="469" height="244" alt="image" src="https://github.com/user-attachments/assets/27b98d20-a33b-4b68-81c3-5a615963648c" />

<img width="468" height="239" alt="image" src="https://github.com/user-attachments/assets/fd05b168-f2a9-4718-962c-6a2d5d2307fe" />














