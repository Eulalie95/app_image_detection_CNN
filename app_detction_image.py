import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.write('''
# Detection des catégories d'images avec CNN
Cette application prédit la catégorie des images en utilisant un modèle de CNN.
''')

st.sidebar.header("Téléchargez votre image")

def user_input():
    uploaded_file = st.sidebar.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Image téléchargée.', use_column_width=True)
        st.write("")
        st.write("Juste un instant...")
        return img
    else:
        return None

image = user_input()

if image is not None:
    # Chargement du modèle de CNN pré-entraîné
    model = load_model('cnn_cifar10_model.h5')

    # Préparation des données pour le modèle CNN
    def prepare_image(img):
        # Convertir l'image en un tableau numpy
        img = np.array(img)
        
        # Redimension de la taille de l'image à 32x32 pixels
        img = cv2.resize(img, (32, 32))
        
        # Normalisation de l'image
        img = (img - mean) / (std + 1e-7)
        
        # Remodeler l'image pour correspondre à la forme attendue par le modèle
        img = img.reshape((1, 32, 32, 3))
        
        return img

    prepared_image = prepare_image(image)


    # Prédiction avec le modèle CNN
    prediction = model.predict(prepared_image)
    predicted_class = np.argmax(prediction, axis=1)

    # Noms des classes CIFAR-10
    cifar10_class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    predicted_class_name = cifar10_class_names[predicted_class[0]]

    st.subheader("La catégorie de l'image est:")
    st.write(predicted_class_name)
else:
    st.write("Veuillez télécharger une image pour la prédiction.")
