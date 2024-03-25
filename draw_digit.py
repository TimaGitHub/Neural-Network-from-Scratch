import PIL
import random
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageDraw,  ImageEnhance
import augmentation
from neural_network import NeuralNetwork
from dataloader import  DataBatcher
import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')

data = np.array(data)

data_with_noise = np.empty(data.shape)

index = 0

for dat in data:
    data_with_noise[index] = np.hstack((dat[0], augmentation.add_some_noise_for_digits(dat[1:])))
    index += 1

data = np.vstack((data, data_with_noise))

np.random.shuffle(data)

test_data = data[0:1000]
train_data = data[1000: 42000]
test_data[:, 1:] = test_data[:, 1:] / 255
train_data[:, 1:] = train_data[:, 1:] / 255

test_batches = DataBatcher(test_data, 64, True)
train_batches = DataBatcher(train_data, 64, True)

test = NeuralNetwork(2, 20, 'classification')
test.prepare(gradient_method = 'sagd', activation_func = 'leaky_relu', seed = None, alpha = 0.3, loss_function = 'cross_entropy_loss')
test.cosmetic(progress_bar = True, loss_display = True, iterations = 1)
test.train(param, digit, 1)

root = Tk()

root.title('Digit Recognizer')
root.resizable(width=False, height=False)

cv = Canvas(root, width=400, height=500, bg='black')
image = Image.new(mode='L', size=(400, 400), )
draw = ImageDraw.Draw(image)

cv.create_line(0, 405, 400, 405, fill='white', width=10)

outline = 'white'


def func(event):
    x2, y2 = event.x + 10, event.y + 10
    x1, y1 = event.x - 10, event.y - 10
    cv.create_rectangle(x1, y1, x2, y2, fill=outline, outline=outline)
    draw.rectangle([(x1, y1), (x2, y2)], width=1, fill=outline, outline=outline)


cv.bind("<B1-Motion>", func)
cv.pack(expand=1, fill=BOTH)


def save():
    filename = 'C:\\Users\\Acer\\Downloads\\image_.png'
    temp_file = image.resize((28, 28), Image.LANCZOS)
    enhancer = ImageEnhance.Sharpness(temp_file)
    factor = 1.5
    temp_file = enhancer.enhance(factor)
    temp_file.save(filename)
    messagebox.showinfo('Saved')


def to_draw():
    global outline
    outline = 'white'


def to_clean():
    global outline
    outline = 'black'


def clean_all():
    cv.create_rectangle(0, 405, 400, 500, fill='black')
    cv.create_rectangle(0, 0, 400, 400, fill='black')
    draw.rectangle([(0, 0), (400, 400)], width=1, fill='black', outline='black')
    global outline
    outline = 'white'


def predict():
    global image
    im = image.copy()

    image_for_prediction = im.resize((28, 28), Image.LANCZOS)
    enhancer = ImageEnhance.Sharpness(image_for_prediction)
    factor = 1.5
    image_for_prediction = enhancer.enhance(factor)

    image_for_prediction = np.array(image_for_prediction).reshape(1, -1) / 255

    cv.create_rectangle(0, 405, 400, 500, fill='black')

    cv.create_text(200, 450, text=f'{test.predict(image_for_prediction).argmax(axis=1)[0]}', fill='white', justify=CENTER,
                   font="Verdana 14")


Button(root, text='').pack(side='right')
DRAW = Button(root, text='DRAW', command=to_draw)
CLEAN = Button(root, text='CLEAN', command=to_clean)
SAVE = Button(root, text='SAVE', command=save)
CLEAN_ALL = Button(root, text='CLEAN_ALL', command=clean_all)
PREDICT = Button(root, text='PREDICT', command=predict)

DRAW.place(x=0, y=505, width=50, height=20)
CLEAN.place(x=70, y=505, width=50, height=20)
CLEAN_ALL.place(x=140, y=505, width=70, height=20)
SAVE.place(x=230, y=505, width=40, height=20)
PREDICT.place(x=290, y=505, width=60, height=20)
root.mainloop()