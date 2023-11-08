import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
from tkinter import OptionMenu, StringVar
from tkinter import ttk

import PIL.Image
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import threading
import os


def display_image(file_path):
    img = Image.open(file_path)
    img = img.resize((200, 200), PIL.Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img


def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        progress_label.config(text="Classifying image...")

        def classify():
            try:
                img = image.load_img(file_path, target_size=(299, 299))
                img = image.img_to_array(img)
                img = preprocess_input(img)
                img = np.expand_dims(img, axis=0)

                predictions = model.predict(img)
                decoded_predictions = keras.applications.inception_v3.decode_predictions(predictions, top=3)[0]

                result_text = "Possible classifications:\n"
                for _, label, probability in decoded_predictions:
                    result_text += f"{label}: {probability:.2%}\n"

                result_label.config(text=result_text)
                display_image(file_path)
                progress_label.config(text="Classification complete")
            except Exception as e:
                progress_label.config(text="Classification error")
                messagebox.showerror("Error", str(e))


        classification_thread = threading.Thread(target=classify)
        classification_thread.start()
    else:
        result_label.config(text="No image selected!")
        image_label.config(image=None)
        progress_label.config(text="")


def clear_results():
    result_label.config(text="")
    image_label.config(image=None)
    progress_label.config(text="")


def load_custom_model():
    custom_model_path = filedialog.askopenfilename()
    if custom_model_path:
        try:
            custom_model = keras.models.load_model(custom_model_path)
            custom_model_var.set(custom_model)
            model_var.set("Custom Model")
            messagebox.showinfo("Success", "Custom model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", "Failed to load custom model")
            print(e)


model = keras.applications.InceptionV3(include_top=True, weights='imagenet')


root = tk.Tk()
root.title("Image Classification")
root.geometry("800x500")

browse_button = Button(root, text="Select an Image", command=classify_image)
browse_button.pack()

clear_button = Button(root, text="Clear Results", command=clear_results)
clear_button.pack()

model_var = StringVar()
model_var.set("InceptionV3")

model_options = OptionMenu(root, model_var, "InceptionV3", "ResNet50", "MobileNetV2", "Custom Model")
model_options.pack()

custom_model_var = StringVar()
custom_model_var.set("Custom Model")

custom_model_button = Button(root, text="Load Custom Model", command=load_custom_model)
custom_model_button.pack()

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="", justify="left")
result_label.pack()

progress_label = Label(root, text="", justify="left")
progress_label.pack()

root.mainloop()
