import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="What_Am_I_Eating",page_icon=":eyes:",layout="wide")
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 2rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 0rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

CLASS_NAMES=['apple_pie',
 'baby_back_ribs',
 'baklava',
 'beef_carpaccio',
 'beef_tartare',
 'beet_salad',
 'beignets',
 'bibimbap',
 'bread_pudding',
 'breakfast_burrito',
 'bruschetta',
 'caesar_salad',
 'cannoli',
 'caprese_salad',
 'carrot_cake',
 'ceviche',
 'cheese_plate',
 'cheesecake',
 'chicken_curry',
 'chicken_quesadilla',
 'chicken_wings',
 'chocolate_cake',
 'chocolate_mousse',
 'churros',
 'clam_chowder',
 'club_sandwich',
 'crab_cakes',
 'creme_brulee',
 'croque_madame',
 'cup_cakes',
 'deviled_eggs',
 'donuts',
 'dumplings',
 'edamame',
 'eggs_benedict',
 'escargots',
 'falafel',
 'filet_mignon',
 'fish_and_chips',
 'foie_gras',
 'french_fries',
 'french_onion_soup',
 'french_toast',
 'fried_calamari',
 'fried_rice',
 'frozen_yogurt',
 'garlic_bread',
 'gnocchi',
 'greek_salad',
 'grilled_cheese_sandwich',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_and_sour_soup',
 'hot_dog',
 'huevos_rancheros',
 'hummus',
 'ice_cream',
 'lasagna',
 'lobster_bisque',
 'lobster_roll_sandwich',
 'macaroni_and_cheese',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'omelette',
 'onion_rings',
 'oysters',
 'pad_thai',
 'paella',
 'pancakes',
 'panna_cotta',
 'peking_duck',
 'pho',
 'pizza',
 'pork_chop',
 'poutine',
 'prime_rib',
 'pulled_pork_sandwich',
 'ramen',
 'ravioli',
 'red_velvet_cake',
 'risotto',
 'samosa',
 'sashimi',
 'scallops',
 'seaweed_salad',
 'shrimp_and_grits',
 'spaghetti_bolognese',
 'spaghetti_carbonara',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'takoyaki',
 'tiramisu',
 'tuna_tartare',
 'waffles']

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('models/fine_tuned_model_with_model_ckpt_wo_mp.h5')
	return model

def load_and_prep(filename,img_shape=224):
    img = tf.cast(filename,tf.float32)
    img = tf.image.resize(img, [img_shape, img_shape])
    return img

def predict(image,model):
    image = load_and_prep(image)
    pred_prob = model.predict(tf.expand_dims(image, axis=0),verbose=0) # make prediction on image with shape [None, 224, 224, 3]
    pred_class = CLASS_NAMES[pred_prob.argmax()]
    pred_class = pred_class.replace('_',' ').capitalize()
    prob_pred_class = tf.reduce_max(pred_prob).numpy()*100
    prob_class_str = "{:.2f}".format(prob_pred_class)
    st.success(f"It is a **{pred_class}** with {prob_class_str}% confidence")

st.markdown("<h1 style='text-align: center;'>What_Am_I_Eating &#127828&#128064;</h1>", unsafe_allow_html=True)
col1,col2 = st.columns([2,1])

with col1:
    st.markdown(":wave: Hello and welcome to the **What_Am_I_Eating** web app. **What_Am_I_Eating** is a web app which categorises food images. It uses a model made using **Tensorflow** (Google's open source library).")
    st.markdown("""
### Introduction
What_Am_I_Eating is an app which classifies the food item present in the image.  The main aim of building this model was to beat the [DeepFood](https://arxiv.org/abs/1606.05675)📄 paper. The app uses a **neural network** made using Tensorflow to classify these images into 101 categories of food items. I made this project while completing the [Zero to mastery Tensorflow course](https://zerotomastery.io/courses/learn-tensorflow/).

### Data
The model is trained on the **[Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)** dataset. This dataset was used in the various papers such as [DeepFood](https://arxiv.org/abs/1606.05675) and [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). The data has 101 food categories and a total of 101,000 images. For each class, there are 250 test images and 750 train images. This is a lot of data. To optimize training I used the Food101 data available in *tensorflow-datasets* library. This made it really easy to use, as the data was already in tensors.

### Model
As said, the model is a neural network made using Tensorflow. The neural network uses *EfficientNetB0* as its backbone. The model has four layers, namely:
* The Input layer: Confirms the inputs to the neural network is in the form of tensor with shape (224,224,3).
* EfficientNetB0: Using the keras API and exploiting the transfer learning, the neural network uses a *EfficientNetB0*. This is a pretrained model on the classical ImageNet dataset. We fine-tune the weights and biases to make the predictions better. Learn more about the architecture of *EfficientNetB0* - [architecture](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)
* GlobalAveragePooling2D layer: This layer takes the average of all the numbers in the previous layer and then condenses it into a (1,3) tensor. Layers such as GlobalAveragePooling2D layer, MaxPooling layers etc., usually come in handy in CNNs as there are a lot of numbers and the output layer may take much time to then predict classes.
* Dense layer: This is used so that we have 1 neuron for each class.
* Activation layer: This layer is used to finally classify the tensors in classes. We have used *softmax* activation function as this is a multi-class classification problem. The activation could've have been integrated with the *Dense* layer itself. We have used this type of structure to use the *mixed_precision* feature of tensorflow and keras.
We then compile the model with *sparse_categorical_crossentropy* loss, *Adam* optimizer and use *accuracy* as a metric.
""",unsafe_allow_html=True)
    st.markdown("[Github](https://github.com/ishandandekar/What_Am_I_Eating)  [Medium](https://medium.com/@ishandandekar/foodvision-3843f38be45e)")

with col2:
    st.markdown("""
    ### Predict on your image!
    """)
    if st.checkbox("Show labels"):
        st.write("There are 101 classes, so these are some 5 labels")
        import random
        st.write(random.sample(CLASS_NAMES,5))
    image = st.file_uploader(label="Upload an image",type=['png','jpg','jpeg'])
    if image is not None:
        st.image(image=image)
        test_image = Image.open(image)
        model = load_model()
        if st.button("Predict"):
            predict(test_image,model)