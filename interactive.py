import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Rectangle
from functools import partial
import tensorflow as tf
from tensorflow import keras
import argparse

def resize_to_closest_acceptible(img):
    original_height, original_width, _ = img.shape

    # Calculate the new dimensions
    new_width = (original_width // 16) * 16
    new_height = (original_height // 16) * 16

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

# Combine an 'L' grayscale image with an 'AB' image, and convert to BGR for display or use
def lab_to_bgr(image_l, image_ab):
    image_l = image_l.reshape((image_l.shape[0], image_l.shape[1], 1))
    image_lab = np.concatenate((image_l, image_ab), axis=2)
    image_lab = image_lab.astype("uint8")
    image_rgb = cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)
    return image_rgb

def colorize_region_event(event, model, original_image):
    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        colorized_image = colorize_region(model, original_image.copy(), x, y)
        update_plot(colorized_image)

def colorize_image(event, model, original_image):
    colorized_image = colorize(model, original_image.copy())
    update_plot(colorized_image)

def colorize(model, region):
    # Convert the region to grayscale
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Make prediction using the colorizer model
    prediction = model.predict(np.expand_dims(region, axis=0))[0]

    colorized_region = lab_to_bgr(region, prediction)

    return colorized_region

def update_plot(img):
    ax.clear()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.draw()

def clear_image(event, original_image):
    update_plot(original_image)

def colorize_region(model, img, x, y):
    x_scale = globals()['x_scale']
    y_scale = globals()['y_scale']

    gray_region = img[max(0, y - 16*y_scale):min(img.shape[0], y + 16*y_scale), max(0, x - 16*x_scale):min(img.shape[1], x + 16*x_scale)]
    colorized_region = colorize(model, gray_region)
    img[max(0, y - 16*y_scale):min(img.shape[0], y + 16*y_scale), max(0, x - 16*x_scale):min(img.shape[1], x + 16*x_scale)] = colorized_region
    
    return img

def update_x_scale(val):
    global x_scale
    x_scale = int(val)

def update_y_scale(val):
    global y_scale
    y_scale = int(val)

def main(FLAGS):
    global ax, x_scale, y_scale

    WEIGHTS_DIRECTORY = './weights/'
    SAMPLES_DIRECTORY = './samples/'

    # Load the pre-trained colorizer model
    #model = keras.models.load_model('unet-batchnorm.h5', compile=False)
    model = keras.models.load_model(WEIGHTS_DIRECTORY + 'unet-batchnorm.h5')

    if FLAGS.start_colored:
        original_image = cv2.imread(SAMPLES_DIRECTORY + FLAGS.input)
    else:
        original_image = cv2.imread(SAMPLES_DIRECTORY + FLAGS.input, cv2.IMREAD_GRAYSCALE)
        original_image = np.expand_dims(original_image, axis=-1)
        original_image = np.concatenate([original_image] * 3, axis=-1)

    original_image = resize_to_closest_acceptible(original_image)

    # Set up matplotlib figure and axes
    fig, ax = plt.subplots()
    if FLAGS.start_colored:
        ax.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY), cmap='gray')

    x_scale = 14
    y_scale = 14

    clear_button = Button(plt.axes([0.8, 0.01, 0.1, 0.05]), 'Clear')
    clear_button.on_clicked(partial(clear_image, original_image=original_image))

    colorize_button = Button(plt.axes([0.1, 0.01, 0.1, 0.05]), 'Colorize')
    colorize_button.on_clicked(partial(colorize_image, model=model, original_image=original_image))

    ax_x_scale = plt.axes([0.35, 0.01, 0.35, 0.02], facecolor='lightgoldenrodyellow')
    ax_y_scale = plt.axes([0.35, 0.04, 0.35, 0.02], facecolor='lightgoldenrodyellow')
    
    s_x_scale = Slider(ax_x_scale, 'X Scale', 1, 50, valinit=x_scale, valstep=1)
    s_x_scale.on_changed(update_x_scale)
    
    s_y_scale = Slider(ax_y_scale, 'Y Scale', 1, 50, valinit=y_scale, valstep=1)
    s_y_scale.on_changed(update_y_scale)

    fig.canvas.mpl_connect('button_press_event', partial(colorize_region_event, model=model, original_image=original_image))

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Interactive Image Colorizer')
    parser.add_argument('--input',
                        type=str, default='flag.jpg',
                        help='Input image file path')
    parser.add_argument('--start_colored',
                        action="store_true",
                        help='Raw input image is shown initially')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)