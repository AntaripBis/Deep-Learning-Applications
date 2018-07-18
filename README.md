# Deep-Learning-Applications
The repository contains different applications of deep learning using the keras and Tensorflow framework. The models have been developed in python. There are 3 applications at present - Speech to Text translation using Recurrent Neural Networks, Video Action Classification using Convolutional Neural Networks and Image Synthesis using Generative Adversarial Networks.

# System Requirements
1. 64-bit Intel or AMD machines
2. At least  8 GB RAM. Neural networks are require memory-intensive computing so 4 GB RAM may not be sufficient or may require certain memory management.
3. Preferably Linux operating system - Ubuntu or Redhat. Windows can be used for deploying Tensorflow but certain libraries used in the analyses may not work properly.
4. GPU (optional) may be used to speed up computation.

# Libraries
1. Tensorflow
2. Keras
3. Librosa
4. OpenCV 

# Datsets used
1. Common Voice dataset for the speech to text translation - https://www.kaggle.com/mozillaorg/common-voice
2. Human Activity Video Datasets for video recognition- https://www.cs.utexas.edu/~chaoyeh/web_action_data/dataset_list.html
3. CIFAR-10 dataset to be used for image generation

# Configuration
The 3 properties files in the config folder are used to set the properties of the applications. Below are some key properties that can be set for the applications.
1. DATA_DIR : Path to the input data
2. MODEL_DIR : Path to where the model will be stored
3. RESULT_DIR : Path to the location where the results of the model will be stored
4. LEARNING_RATE : Learning rate for the optimizers used in the model
5. BATCH_SIZE : number of instances to be considered in each batch while training
6. LOAD_PREVIOUS_MODEL_FLAG : it can take true or false value, if set to true, it will load the model already trained from the MODEL_DIR location
7. TRAINING_RATIO : it specifies the ratio in which the data has to be split into train and test set
