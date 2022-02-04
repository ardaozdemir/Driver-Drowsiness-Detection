from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
#from imutils.video import VideoStream
import tensorflow as tf
import numpy as np
#import imutils
import time
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime

#Loading CNN model and cascade classifier files
eye_model = tf.keras.models.load_model('eye_detector.h5')
face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

#Starting video capturing
capture = cv2.VideoCapture(1)
if not capture.isOpened():
    capture = cv2.VideoCapture(0)
if not capture.isOpened():
    raise IOError("Cannot access to the webcam")

counter = 0

#Starting detection
while True:
    #To calculate duration of the process datetime imported.
    start_time = datetime.now()

    #Capturing image and detecting eyes
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_detect.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in eyes:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        new_eyes = eye_detect.detectMultiScale(roi_gray)
        if len(new_eyes) == 0:
            print("No eyes detected")
        else:
            for (ex, ey, ew, eh) in new_eyes:
                eyes_roi = roi_color[ey: ey + eh, ex:ex + ew]

    #Converting eyes to make ready for prediction
    final_eye = cv2.resize(eyes_roi, (224, 224))
    final_eye = np.expand_dims(final_eye, axis=0)
    final_eye = final_eye / 255.0

    #Predicting the eye status of driver
    Predictions = eye_model.predict(final_eye)

    #Checking if driver's eye open or close. Also increasing counter if it close
    if (Predictions[0][0] >= Predictions[0][1]):
        status = "open"
        if (counter > 0):
            counter -= 1
    else:
        status = "close"
        counter +=1
        #If it is closed for 3 seconds 
        if (counter > 15):
            status = "Drowsy!!"
            
    #Detecting face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(face_detect.empty())
    faces = face_detect.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #Priting status
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, status, (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4)
    cv2.imshow('test', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

    #Priting the duration of detection
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

capture.release()
cv2.destroyAllWindows()