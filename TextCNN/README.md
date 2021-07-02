# Text-CNN for sentimental analysis

Utilize Text-CNN to classify positive/negtive movie reviews.

## Database: A Brief Introduction

The database has been preprocessed and contains 4 parts:
* _Training set_: 19998 movie reviews, pos: 9999, neg: 9999
* _Validation set_: 5629 movie reviews, pos: 2912, neg: 2817
* _Testing set_: 369 movie reviews, pos: 182, neg: 187
* _Word2vec_: pretrained, of size (58954, 50)

## Text-CNN

Similar to traditional CNNs, Text-CNN contains an embedding layer, convolution layers,  pooling layers and fully-connected layers. Refer to the original paper for details:

* Y. Kim, “[Convolutional Neural Networks for Sentence Classification by ](https://arxiv.org/pdf/1408.5882.pdf),” Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014. 
