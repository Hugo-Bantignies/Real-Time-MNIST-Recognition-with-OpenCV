import cv2
import numpy as np
from keras.models import load_model

#Threshold for binary_inv
threshold = 100

#Size and position of the region for the digits
x, y, h, w = (200, 200, 125, 125)

#Get the camera
cap = cv2.VideoCapture(0)

#Load the model (LeNet5)
net5 = load_model("net5.h5")

while(True):
    # Camera
    ret, frame = cap.read()

    # Region of interest for the digit
    region = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    #Rectangle on the frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #Data preparation (from the region size to 28 x 28)
    resized_digit = cv2.resize(gray, (28,28))
    data = np.array(resized_digit)/255 #Normalization
    data = data.reshape(1,28,28,1)

    #Prediction
    prediction = net5.predict(data)
    predicted_class = np.argmax(prediction, axis=-1) 
    
    #Display the prediction
    cv2.putText(frame, str(predicted_class[0]), (x-30, y+20), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,255,0), 2)

    #Display actions
    cv2.putText(frame,"'a' : Save image",(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    # Display camera + region and frame
    cv2.imshow('digit',gray)
    cv2.imshow('camera', frame)


    #Actions with keys
    key = cv2.waitKey(1) & 0xFF

    #Save the image within the frame
    if key == ord('a'):
        cv2.imwrite("res.png",resized_digit)

    #Exit the application
    elif key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()