
# Deep Learning for Breast Cancer Classification and CIFAR-10 Dataset

## Overview

This repository contains the implementation of VGG19 and ResNet50 models for image classification tasks. The project focuses on two datasets: the INbreast dataset, which consists of mammography images for breast cancer classification, and the CIFAR-10 dataset (for learning purposes), a widely-used dataset for object recognition.

## Datasets

### INbreast Dataset
The INbreast dataset includes 410 DICOM mammography images categorized using the BI-RADS system, which helps classify breast cancer likelihood. The dataset was used to train and evaluate the ResNet50 model.

### CIFAR-10 Dataset
The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. This dataset was used to initially test the models and evaluate their performance.

## Models

### VGG19
- **Dataset**: CIFAR-10
- **Accuracy**: 90.55%
- **Details**: VGG19 showed high accuracy in categories like cars and horses but struggled with cats and birds.

### ResNet50
- **Dataset**: CIFAR-10 and INbreast
- **Accuracy on CIFAR-10**: 86.94%
- **Accuracy on INbreast**: 81.27%
- **Details**: ResNet50 performed well on CIFAR-10, especially for ships and trucks. On the INbreast dataset, it showed varying performance across different BI-RADS categories.

## Results

The models showed promising results, with VGG19 achieving 90.55% accuracy on CIFAR-10 and ResNet50 reaching 81.27% on the INbreast dataset. The models' performance varied across different categories, indicating areas for further optimization.

## Presentations

Two presentations related to this project are available in the `presentations` folder:
1. Artificial Imaging in Breast Imaging. 
2. Deep Learning to Improve Breast Cancer Detection on Screening Mammography.

## Acknowledgments

- **Internship Location**: IISER Thiruvananthapuram
- **Department**: School of Data Science
- **Guide**: Dr. Raji Susan Mathew

## References

1. [Artificial Intelligence in Breast Imaging](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10582610/)
2. [INbreast Database](https://www.sciencedirect.com/science/article/abs/pii/S107663321100451X)
3. [Deep Learning to Improve Breast Cancer Detection](https://www.nature.com/articles/s41598-019-48995-4)
4. [Review of Deep Learning Concepts](https://doi.org/10.1186/s40537-021-00444-8)
