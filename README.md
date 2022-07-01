# What_Am_I_Eating
<p align="center">
    <img width=450 height=300 src="assets/meme.jpeg">
</p>


Hello and welcome, What_Am_I_Eating is my first computer vision project (get the meme now?).  
Check out the [Medium](https://medium.com/@ishandandekar/foodvision-3843f38be45e) article as a supplement to this README. The article describes the project better.

## About
What_Am_I_Eating is an app which classifies the food item present in the image.  The main aim of building this model was to beat the [DeepFood](https://arxiv.org/abs/1606.05675)📄 paper.The app uses a neural network to classify these images into 101 categories of food items.

## Data
Data used to train the image is available on [Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101). Although I used the dataset available in the [tensorflow datasets catalog](https://www.tensorflow.org/datasets/catalog/food101). The advantage of using the `tensorflow_datasets` module is that the data is already in tensors. Hence, there is little to no requirement of preprocessing.

## Model
The model uses transfer learning to use the `EfficientNetB0` architecture under the hood. The model has four layers, namely:
* The Input layer: Confirms the inputs to the neural network is in the form of tensor with shape (224,224,3).
* EfficientNetB0: Using the keras API and exploiting the transfer learning, the neural network uses a `EfficientNetB0`. This is a pretrained model on the classical ImageNet dataset. We fine-tune the weights and biases to make the predictions better. Learn more about the architecture of `EfficientNetB0` - [architecture](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)
* GlobalAveragePooling2D layer: This layer takes the average of all the numbers in the previous layer and then condenses it into a (1,3) tensor. Layers such as GlobalAveragePooling2D layer, MaxPooling layers etc., usually come in handy in CNNs as there are a lot of numbers and the output layer may take much time to then predict classes.
* Dense layer: This is used so that we have 1 neuron for each class.
* Activation layer: This layer is used to finally classify the tensors in classes. We have used `softmax` activation function as this is a multi-class classification problem. The activation could've have been integrated with the `Dense` layer itself. We have used this type of structure to use the `mixed_precision` feature of tensorflow and keras.  
We then compile the model with `sparse_categorical_crossentropy` loss, `Adam` optimizer and use `accuracy` as a metric.

## App
The web app is made using Streamlit. I chose this framework as it is very easy to make web apps with it and then deploy it. The website is accompanied by the model (in .h5 file format). This reduces the time taken to predict. The website also displays a probability graph after the prediction. This graph shows the probability of the image being of that class (only top 5 classes).