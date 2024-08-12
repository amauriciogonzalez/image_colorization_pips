import cv2
import time
import argparse
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import os

# Prepares an image for prediction
def resize_to_closest_acceptible(img):
    original_height, original_width = img.shape

    # Calculate the new dimensions
    new_width = (original_width // 16) * 16
    new_height = (original_height // 16) * 16

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


def colorize_frame(model, frame):
    # Convert the resized frame to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame to a suitable size for colorization
    frame = resize_to_closest_acceptible(frame)

    prediction = model.predict(np.expand_dims(frame, axis=0))[0]

    # Colorize the grayscale frame
    colorized_frame = lab_to_bgr(frame, prediction)
    
    return colorized_frame


# Combine an 'L' grayscale image with an 'AB' image, and convert to BGR for display or use
def lab_to_bgr(image_l, image_ab):
    image_l = image_l.reshape((image_l.shape[0], image_l.shape[1], 1))
    image_lab = np.concatenate((image_l, image_ab), axis=2)
    image_lab = image_lab.astype("uint8")
    image_bgr = cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)
    return image_bgr
    

# This function determines whether or not the inputted file is a video file.
def is_video_file(file_path):
    # If we're using the webcam, return false. No need to calculate FPS.
    if not file_path:
        return False
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'] 
    _, extension = os.path.splitext(file_path)
    is_video = extension.lower() in video_extensions
    return is_video


def main(args):
    MODEL_FOLDER_NAME = 'weights'
    SAMPLE_FOLDER_NAME = 'samples'

    # Load one of the four desired gender/age predictors
    model = keras.models.load_model(f'./{MODEL_FOLDER_NAME}/unet-batchnorm.h5', compile=False)

            
        
    # An image or video is read.
    if args.input:
        input_file = f'./{SAMPLE_FOLDER_NAME}/' + args.input
    # Otherwise, we consider the input file as the webcam.
    else:
        input_file = 0

    is_video = is_video_file(input_file)
    cap = cv2.VideoCapture(input_file)

    # If a video is detected, start counting frames to calculate FPS
    if is_video:
        start_time = time.time()
        frame_count = 0

    cv2.namedWindow('Colorization', cv2.WINDOW_NORMAL)

    # Here, each frame is cycled through.
    while cv2.waitKey(1) < 0:
        
        # A frame is initially read.
        hasFrame, frame = cap.read()
        if not hasFrame:
            if is_video:
                cv2.waitKey(1)
            else:
                cv2.waitKey(0)
            break

        resized_frame = cv2.resize(frame, (224, 224))

        # Colorize the frame
        colorized_frame = colorize_frame(model, resized_frame)

        #colorized_frame = cv2.resize(colorized_resized_frame, (frame.shape[1], frame.shape[0]))

        # Display the colorized frame
        cv2.imshow('Colorization', colorized_frame)

        # Frames are counted to display FPS
        if is_video:
            frame_count += 1
        


    # Display FPS for videos
    if is_video:
        fps = frame_count / (time.time() - start_time)
        print(f"Average FPS: {fps :.2f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real Time Colorization')
    parser.add_argument('--input',
                        help='A file path to an input image or video file. Leave blank to use webcam.')
    args = parser.parse_args()

    main(args)