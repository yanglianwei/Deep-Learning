# A simple Convolutional Neural Network (CNN) for classifying the MNIST database

A simple implementation of a CNN to classify MNIST digits dataset.

## MNIST Database: A Brief Introduction

The MNIST database is a dataset of *handwritten digits*. It mainly contains:
* A training set of 60,000 examples
* A test set of 10,000 examples

The digits have been **size-normalized** (28x28 pixels) and centered in a **fixed-size single channel** (Black & White) image.

The dataset is available online from Yann LeCun's personal website: [the MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## CNN structure: based on LeNet-5

The structure used in this simple implementation is depicted as follows:

![CNN_structure](https://github.com/ML1998/DeepLearningExamples/blob/main/MNIST/CNNstruct.png)

## Implementation: CNN for MNIST classifying

### 1. Tensorflow Implementation

see [mnist_cnn.ipynb](https://github.com/MingyuL98/Deep-Learning/blob/main/MNIST/mnist_cnn.ipynb)

#### 1.1 Prepare MNIST data

* Data preprocessing

      img_train = extract_images(DATA_PATH + TRAIN_IMAGES_PATH)
      label_train = extract_labels(DATA_PATH + TRAIN_LABELS_PATH)
      img_test = extract_images(DATA_PATH + TEST_IMAGES_PATH)
      label_test = extract_labels(DATA_PATH + TEST_LABELS_PATH)

* Seperate validation set (optional)

      VAL_SET_SIZE = 6000 

* Convert to float type
* Shuffle and batch data

      train_data = tf.data.Dataset.from_tensor_slices((img_train, label_train))
      train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

#### 1.2 Construct CNN
        
* CNN Structure
    * Conv Layer 1: 5x5 conv, 1 input, 32 filters (MNIST has 1 color channel only).

    * Conv Layer 2: 5x5 conv, 32 inputs, 64 filters.

    * FC Layer 1: 7*7*64 inputs, 512 units.
    * FC Out Layer: 512 inputs, 10 units (total number of classes)

* Build the conv layer and maxpooling layer

      def conv_2d(x, weights, bias, strides=1):
          x = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding='SAME')
          x = tf.nn.bias_add(x, bias)
          return tf.nn.relu(x)


      def maxpool_2d(x, k=2):
          return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
#### 1.3 Model Training

* Compute cross-entropy loss
          
      tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))
    
* Optimizer: Adam

      optimizer = tf.optimizers.Adam(learning_rate)
* Compute gradients and update W and b following gradients
    
      trainable_variables = list(weights.values()) + list(biases.values())
      gradients = g.gradient(loss, trainable_variables)
      optimizer.apply_gradients(zip(gradients, trainable_variables))

### 2. Pytorch Implementation

see [mnist_cnn_torch.ipynb](https://github.com/MingyuL98/Deep-Learning/blob/main/MNIST/mnist_cnn_torch.ipynb)
