# YOLO-P6-Based-Multiclass-Malaria-Parasite-Detector-for-Multi-Scale-Targets
Multiclass object detection model for identifying malaria parasites in blood smear images, designed to support fast and accurate diagnosis in resource-limited settings.


<div style="display:flex; text-align:center; justify-content:center;"> 
    <img src="assets/intro_1.png" alt="introduction_image" width="300">
    <img src="assets/intro_3.png" alt="introduction_image" width="200">
    <img src="assets/intro_2.png" alt="introduction_image" width="300">
</div>


## Overview
Malaria remains one of the world’s most deadly infectious diseases, causing hundreds of thousands of deaths each year—particularly among children under five and pregnant women in Africa. Early and accurate diagnosis is critical for effective treatment and disease management. However, traditional diagnostic methods such as microscopic examination of blood slides are resource-intensive and depend on skilled personnel, which are often unavailable in rural or low-resource regions.

This project presents a machine learning solution to automate malaria detection using object detection and classification techniques. By leveraging computer vision, we aim to support scalable and efficient diagnostic workflows that can be deployed in real-world healthcare settings across Africa.

## Objective
The goal of this challenge is to develop a multiclass object detection and classification model that can:

Accurately detect malaria parasites in microscopic blood slide images

Identify the trophozoite stage of malaria infection

Classify infected vs. uninfected red blood cells

This solution is tailored for low-resource environments and could play a key role in large-scale screening programs, early warning systems for outbreak detection, and relieving diagnostic workload from healthcare professionals.

## Dataset
The dataset which can be found on [Zindi](https://zindi.africa/competitions/ghana-crop-disease-detection-challenge/data) consists of high-resolution blood smear images captured by placing a smartphone camera over a microscope eyepiece, simulating a realistic, low-cost diagnostic setup. For each image, the dataset includes:

i. The slide ID from which the image was captured

ii. Microscope stage micrometer readings

This dataset was curated to improve the generalization ability of malaria detection models and complements existing malaria microscopy datasets. It is particularly valuable for developing models that perform well across varied field conditions, such as those in communities like Uganda.

## Model

### YOLOv8-p6: A real-time object detection model designed for multi-scale targets.

![YOLOv8](assets/YOLOv8.webp)


## Evaluation Metrics

To compare the models fairly, we use the following evaluation metrics:

__Mean Average Precision (mAP50):__ Measures overall detection accuracy at an Intersection over Union (IoU) threshold of 0.5.

__Confusion Matrix:__ A confusion matrix for object detection and classification is a performance measurement tool that evaluates the accuracy of a model in predicting both the presence of objects (localization) and their correct class labels (classification).

## Results
| Model   | mAP@50 | mAP@50-95 |
|---------|--------|----------|
| YOLOv8-p6 |   |     |
