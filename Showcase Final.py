###############################################################################
###############################################################################
###############################################################################
import os
import nltk
import keras
import os.path
import numpy as np
import pandas as pd
import string as st
import tkinter as tk
import matplotlib.pyplot as plt
from string import digits
from numpy.random import seed
from nltk.corpus import stopwords
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG19
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers.embeddings import Embedding
from keras.initializers import glorot_uniform
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Input, Dropout, LSTM, Activation, GRU, Flatten
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image

base_dir = 'D:/cancer-cnn'
base_image_dir = os.path.join(base_dir, '')
seed(1)
img_size = 96

conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
model = models.Sequential()
model.add(conv_base)
model.add(BatchNormalization())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax'))
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block4_conv2':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
              
model.compile(loss='binary_crossentropy', 
              optimizer=optimizers.SGD(lr=0.001, nesterov = True), 
              metrics=['acc']) 

model.load_weights('image.h5')
from keras.preprocessing.image import load_img, img_to_array
def image_process(img):
    image = load_img(img, target_size =(img_size, img_size))
    image_array = img_to_array(image)/255
    return image_array


def prediction(model, img_array, items_l):
    prob = model.predict(img_array.reshape(1,img_size,img_size,3))
    pro_df = pd.DataFrame(prob, columns = items_l)
    result = items_l[np.argmax(prob)]
    return pro_df, result

###############################################################################
###############################################################################
###############################################################################
##################################SENT ANALYSIS################################
###############################################################################
###############################################################################
###############################################################################

df = pd.read_csv('rnn-train.csv')
df1 = df.loc[:,['reviewText', 'overall']]
df2 = df1.dropna()
df2 = df2.head(20000)
x_train = df2.loc[:,'reviewText'].values
y_train = df2.loc[:,'overall'].values

max_len = -1
for example in x_train:
    if len(example.split()) > max_len:
        max_len = len(example.split())

if min(y_train) != 0:
    for i in range(y_train.shape[0]):
        y_train[i] -= 1

en_stops = set(stopwords.words('english'))
en_stops = en_stops.union('I')
x_train2 = []
string = ""
for k in range(0, x_train.shape[0]):
    x_train[k] = x_train[k].translate(str.maketrans('', '', st.punctuation))
    holderlist = x_train[k].split(" ")
    for word in holderlist:
        if word in en_stops:
            holderlist.remove(word)           
        string2 = ' '.join(holderlist)
    x_train2.append(string2)
x_train2 = np.asarray(x_train2)

texts = x_train2[:]
labels = y_train[:]
maxlen = 100 
training_samples = 15000 
validation_samples = 5000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
y_train = np.eye(5)[y_train.reshape(-1)]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
y_val = np.eye(5)[y_val.reshape(-1)]

import os
glove_dir = 'D:/'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 300
embedding_matrix = np.zeros((10000, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

modelx = Sequential()
modelx.add(Embedding(max_words, embedding_dim, input_length=maxlen))
modelx.add(GRU(64,return_sequences=True))
modelx.add(Dropout(0.3))
modelx.add(GRU(32))
modelx.add(Dropout(0.3))
modelx.add(Dense(5, activation='softmax'))
modelx.layers[0].set_weights([embedding_matrix])
modelx.layers[0].trainable = False

modelx.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['acc'])
modelx.load_weights('sent.h5')

def text_to_rating():
    text = input()
    x_input = np.array([text])
    sequences=tokenizer.texts_to_sequences(x_input)
    data = pad_sequences(sequences,maxlen=100)
    word_index=tokenizer.word_index
    textpred = np.argmax(modelx.predict(data))
    print(int(textpred) + 1)

###############################################################################
###############################################################################
###############################################################################
##################################TKINTER SETUP################################
###############################################################################
###############################################################################
###############################################################################
    
window = tk.Tk()
window.title("Welcome to Ngee Ann Open House 2020!")
window.resizable(False, False)
#window.tk.call('wm', 'iconphoto', window._w, tk.PhotoImage(file = 'py.png'))
w = 750 #width for the Tk root
h = 700 # height for the Tk root
ws = window.winfo_screenwidth() # width of the screen
hs = window.winfo_screenheight() # height of the screen

# calculate x and y coordinates for the Tk root window
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)

window.geometry('%dx%d+%d+%d' % (w, h, x, y))
tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab_control.add(tab1, text = '   Image Recognition   ')
tab_control.pack(expand = 1, fill = 'both')
tab2 = ttk.Frame(tab_control)
tab_control.add(tab2, text = '   Sentiment Analysis   ')
tab_control.pack(expand = 1, fill = 'both')

## Tab 1: Image Rec ##
lbl1_1 = Label(tab1, text = 'Image Recognition Model', padx = 20, pady = 20, font = ("Arial", 20), anchor = "center").pack()
lbl1_2_text = "This model uses the Keras library and TensorFlow backend for Python to create a Convulated Neural Network (CNN) that can reliably recognize a picture, given a set of fixed outputs. It is trained on images that are tagged with the outputs, and through complex statistical and computing methods, they output a prediction. \n\n Select an image and try it for yourself! \n\n"
lbl1_2 = Label(tab1, text = lbl1_2_text, font = ("Arial", 12), wraplength = 480).pack()
def btn1clicked():
    picsize = 300,300
    file = filedialog.askopenfilename()
    
    a = Image.open(str(file))
    b = a.resize((300, 300), Image.ANTIALIAS)
    img0 = ImageTk.PhotoImage(b)
    canvas.create_image(0, 0, anchor = "nw", image=img0)

    base_image_dir = os.path.join(base_dir, 'images')
    for root, dirs, files in os.walk(base_image_dir):
        subfolders = dirs
        break   
    label_file = os.path.join(base_dir, '2.txt')
    with open(label_file, 'r') as fl:
        xa = fl.readlines()  
    food_list =[]
    for item in xa:
        food_list.append(item.strip('\n'))

    img_array = image_process(file)
    prob_df, result = prediction(model, img_array, sorted(food_list))
    
    lbltext = "Prediction: " + result.title().replace("_", " ") + "\n\n" + "Certainty: " + str(round(prob_df.iloc[0][np.argmax(prob_df.values)] * 100, 5)) + "%"
    lbl1_3.configure(text = lbltext)
    window.canvas.update_idletasks()
    
btn1 = Button(tab1, text="Select Image", command = btn1clicked).pack()

canvas = Canvas(tab1, width = 300, height = 300)  
canvas.pack(side=tk.LEFT, padx=(50, 0))  
img1 = ImageTk.PhotoImage(Image.open("download.jpg"))  
canvas.create_image(2, 2, anchor = "nw", image=img1)

lbl1_3 = Label(tab1, text = '', padx = 20, pady = 20, font = ("Arial", 14), anchor = "center")
lbl1_3.pack(side=tk.RIGHT, padx=(0, 50))

    
## Tab 2: Sent Analysis ##
lbl2_1 = Label(tab2, text = 'Sentiment Analysis Model', padx = 20, pady = 20, font = ("Arial", 20), anchor = "center").pack()
lbl2_2_text = "This model uses the NTLK and Keras libraries (with the TensorFlow backend) for Python to create a Recurrent Neural Network (RNN) model that can analyze feelings from text, given a set of fixed outputs. It is trained on reviews from the Amazon phones dataset, and through complex statistical and computing methods, they output a prediction. \n\n Choose a file and try it out! \n\n"
lbl2_2 = Label(tab2, text = lbl2_2_text, font = ("Arial", 12), wraplength = 480).pack()

#txt = Entry(tab2, width = 80)
#txt.pack()

def text_to_rating(intext):
    text = intext
    x_input = np.array([text])
    sequences=tokenizer.texts_to_sequences(x_input)
    data = pad_sequences(sequences,maxlen=100)
    word_index=tokenizer.word_index
    textpred = np.argmax(modelx.predict(data))
    return int(textpred) + 1

from tkinter import messagebox

def btn2clicked():
    file_csv = filedialog.askopenfilename()
    dfhold = pd.read_csv(file_csv).head(500)
    en_stops = set(stopwords.words('english'))
    for k in range(len(dfhold["reviewText"])):
        dfhold.iloc[k][0] = dfhold.iloc[k][0].translate(str.maketrans('', '', st.punctuation))
        holderlist = dfhold.iloc[k][0].split(" ")
        for word in holderlist:
            if word in en_stops:
                holderlist.remove(word)           
            string2 = ' '.join(holderlist)
        dfhold.iloc[k][0] = string2
    dfhold["rating"] = np.nan
    
    for i in range(len(dfhold["reviewText"])):
        rating = text_to_rating(dfhold.at[i, 'reviewText'])
        dfhold.at[i, 'rating'] = rating
        
    #path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
    dfhold.to_csv(r'C:/Users/abhay/OneDrive/Desktop/out.csv', index = False)
    messagebox.showinfo('Done', 'File saved on your desktop!')
    #aa = Image.open(str(file2))
    #bb = aa.resize((100, 100), Image.ANTIALIAS)
    #img4 = ImageTk.PhotoImage(bb)
    #canvas2.create_image(0, 0, anchor = "nw", image=img4)
    window.canvas2.update_idletasks()

lbl2_3 = Label(tab2, text = '', font = ("Arial", 12), wraplength = 480).pack()
btn2 = Button(tab2, text="Analyze", command = btn2clicked).pack()
lbl2_4 = Label(tab2, text = '', font = ("Arial", 12), wraplength = 480).pack()
canvas2 = Canvas(tab2, width = 100, height = 100)  
canvas2.pack()  
img2 = ImageTk.PhotoImage(Image.open("download.jpg"))  
canvas2.create_image(2, 2, anchor = "nw", image=img2)

window.mainloop()