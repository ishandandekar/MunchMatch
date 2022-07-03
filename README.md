# What_Am_I_Eating
<p align="center">
    <img width=300 height=168.75 src="assets/meme.jpeg">
</p>


Hello and welcome to What_Am_I_Eating. What_Am_I_Eating is my first computer vision project (get the meme now? the meme has vision in it and vision is also a computer, so "computer vision").
Check out the [Medium](https://medium.com/@ishandandekar/foodvision-3843f38be45e) article as a supplement to this README. The article describes the project better.  

## Introduction
What_Am_I_Eating is an app which classifies the food item present in the image.  The main aim of building this model was to beat the [DeepFood](https://arxiv.org/abs/1606.05675)📄 paper.The app uses a neural network to classify these images into 101 categories of food items. I made this project while completing the [Zero to mastery Tensorflow course](https://zerotomastery.io/courses/learn-tensorflow/).  

While doing this project I learned the various techniques and processes. I learned how to get preprocessed data using `tensorflow-datasets` library,how we can leverage other pre-trained models using **transfer learning**, how we can fine-tune these pre-trained models using *fine tuning* method.

## Data
The model is trained on the **[Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)** dataset. This dataset was used in the various papers such as [DeepFood](https://arxiv.org/abs/1606.05675) and [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). The data has 101 food categories and a total of 101,000 images. For each class, there are 250 test images and 750 train images. This is a lot of data. To optimize training I used the Food101 data available in *tensorflow-datasets* library. This made it really easy to use, as the data was already in tensors.
> :bulb:**Note:** You can also find the dataset on **Kaggle** ([link](https://www.kaggle.com/datasets/dansbecker/food-101))

## Model
The model uses transfer learning to use the `EfficientNetB0` architecture under the hood. The model has four layers, namely:
* The Input layer: This layer confirms that the inputs to the neural network is in the form of tensor with shape (224,224,3).
* EfficientNetB0: Using the keras API and exploiting the transfer learning, the neural network uses a `EfficientNetB0`. This is a pretrained model on the classical ImageNet dataset. We fine-tune the weights and biases to make the predictions better. Learn more about the architecture of `EfficientNetB0` - [architecture](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)
* GlobalAveragePooling2D layer: This layer takes the average of all the numbers in the previous layer and then condenses it into a (1,3) tensor. Layers such as GlobalAveragePooling2D layer, MaxPooling layers etc., usually come in handy in CNNs as there are a lot of numbers and the output layer may take much time to then predict classes.
* Dense layer: This is used so that we have 1 neuron for each class.
* Activation layer: This layer is used to finally classify the tensors in classes. We have used `softmax` activation function as this is a multi-class classification problem. The activation could've have been integrated with the `Dense` layer itself. We have used this type of structure to use the `mixed_precision` feature of tensorflow and keras.
We then compile the model with `sparse_categorical_crossentropy` loss, `Adam` optimizer and use `accuracy` as a metric.  

The model got around **78% accuracy** on the test set.

## App
The web app is made using Streamlit. I chose this framework as it is very easy to make web apps with it and then deploy it. The website is accompanied by the model (in .h5 file format). This reduces the time taken to predict. The website also displays a probability graph after the prediction. This graph shows the probability of the image being of that class (only top 5 classes).

## Contents of the repository
* assets: This folder contains images and other unecessary files, which are either required by the app or the Medium article
* fine_tuned_model.h5: This is the Tensorflow model (in .h5 format) which the app uses to classify the images
* app.py: The web app made using streamlit library. This is the python script use to make the web app.
* requirements.txt: As the convention goes, this file has information about the `conda` environment used to create this project.