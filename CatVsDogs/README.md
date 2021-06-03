# A Convolutional Neural Network (CNN) for identifying Cats VS Dogs

A simple implementation of CNN to identify Cats Vs Dogs in real-world images.

## Database: A Brief Introduction

The original Cat VS Dogs database is provided by Kaggle for competition. From more details, go to the official website for this competition: [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)

Here I divide the orginal training set into a smaller training set and a validation set, which contains:
* A training set of 20,000 examples: 10,000 cats and 10,000 dogs
* A validation set of 5,000 examples: 2,500 cats and 2,500 dogs

The images are all **real-world images** which have not been normalized or prepocessed. They are all **3-channel** colored RGB images.

The dir structure is depicted as follows:

    dataset(cats vs dogs)
    | -- train
        | -- dogs (dog.x.jpg)
        | -- cats (cat.x.jpg)
    | -- validation
        | -- dogs (dog.x.jpg)
        | -- cats (cat.x.jpg)

Considering the time and resource limitation, in this implementation, I used
* a trainig set of 4800 images
* a validation set of 1600 images

## CNN structure

The structure used in this simple implementation is depicted as follows:

<img src="https://github.com/ML1998/DeepLearningExamples/blob/main/CatVsDogs/CNNstruct.png" width = "400" height = "500" alt="" align=center />

## Implementation: 

see the notebook file: https://github.com/ML1998/DeepLearningExamples/blob/main/CatVsDogs/catvsdogs_cnn.ipynb