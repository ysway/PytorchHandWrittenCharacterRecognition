# COMPSYS 302 Python Project

## Abstract
The aim of this project is to design a handwritten character recognition (HCR) of English alphabets and number digits using Neural Network. The project is developed in Pytorch using python and trained from the EMNIST dataset. The main approach for the project is classification and segmentation using Convolutional Neural Network(CNN). Recurrent Neural Network long, Inception V4 and Inception Resnet V2 has also been implemented during the training process.

## About EMNIST
Extended MNIST(EMNIST) is a database that contains a large group of handwriting image samples which include lowercase and uppercase handwritten letters and numbers. It is derived from NIST Special Database 19 and converted to a 28x28 pixel gray scale image which matches the structure of MNIST dataset.

EMNIST contains 6 different splits: by class, by merge,balanced, letters, digits and mnist and the config of each split is (NIST. (2017)):
* EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
* EMNIST ByMerge: 814,255 characters. 47 unbalanced classes.
* EMNIST Balanced:  131,600 characters. 47 balanced classes.
* EMNIST Letters: 145,600 characters. 26 balanced classes.
* EMNIST Digits: 280,000 characters. 10 balanced classes.
* EMNIST MNIST: 70,000 characters. 10 balanced classes.

## Models
* Convolutional Neural Network(CNN)
* LSTM of Recurrent Neural Network(RNN) 
* Inception V4
* Inception  Resnet V2

## Dependencies
* cffi>=1.14.0
* cmake>=3.17.2
* ctypes>=1.0.2
* cycler>=0.10.0
* furture>=0.17.1
* joblib>=0.14.1
* kiwisolver>=1.2.0
* matplotlib>=3.2.1
* mkl-include>=2019.0
* mkl>=2019.0
* ninja>=1.7.2
* numpy>=1.15.3
* pillow>=5.3.0
* pyparsing>=2.4.7
* python-dateutil>=2.8.1
* pyyaml>=3.13
* scikit-learn>=0.22.2.post1
* setuptools>=46.1.3
* six>=1.14.0
* sklearn>=0.0
* torch>=1.5.0
* torchvision>=0.6.0

# Authors
* Siwei Yang
* Kun Wang
# License
<p align="left">
    <a href="https://www.gnu.org/licenses/agpl-3.0.html" target="_blank">
    <img width="15%" src="https://www.gnu.org/graphics/agplv3-with-text-162x68.png" style="max-width:100%;">
    </a>
</p>
