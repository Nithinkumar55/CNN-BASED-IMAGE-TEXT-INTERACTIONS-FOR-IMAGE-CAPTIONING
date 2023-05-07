from io import BytesIO
import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image

from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model
import matplotlib
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

BASE_DIR = './dataset/'
WORKING_DIR = './dataset/working'


@st.cache_resource
def load_models():
    with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
        next(f)
        captions_doc = f.read()
    mapping = {}
    for line in captions_doc.split('\n'):
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        caption = " ".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)

    def clean(map):
        for k, captions in map.items():
            for i in range(len(captions)):
                # take one caption at a time
                capt = captions[i]
                # preprocessing steps
                # convert to lowercase
                capt = capt.lower()
                # delete digits, special chars, etc.,
                capt = capt.replace('[^A-Za-z]', '')
                # delete additional spaces
                capt = capt.replace("\s+", ' ')
                # add start and end tags to the caption
                capt = 'startseq ' + " ".join([word for word in capt.split() if len(word) > 1]) + ' endseq'
                captions[i] = capt

    clean(mapping)

    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
    max_length = max(len(caption.split()) for caption in all_captions)
    max_length
    # tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    model_trained = load_model("best_model.h5")

    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    return features, tokenizer, model_trained, vgg_model, max_length


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text


def output(vgg_model, feature, tokenizer, max_length, model, file):
    if file and predict:
        ima = file.read()
        images = Image.open(BytesIO(ima))
        ima = images.resize((224, 224))
        ima = img_to_array(ima)
        ima = ima.reshape((1, ima.shape[0], ima.shape[1], ima.shape[2]))
        # preprocess image for vgg
        ima = preprocess_input(ima)
        # extract features
        feature = vgg_model.predict(ima, verbose=0)
        output_caption = predict_caption(model, feature, tokenizer, max_length)
        st.image(images)
        st.write(output_caption)
    else:
        st.write("Image not uploaded successfully")

st.title("Major Project - Caption generation from the images¸")
file = st.file_uploader("Upload image to predict caption", type=["jpg", "jpeg", "png"])
features, tokenizer, model_trained, vgg_model, max_length = load_models()
predict = st.button("predict")
if predict:
    output(vgg_model, features, tokenizer, max_length, model_trained, file)
