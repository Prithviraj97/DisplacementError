import cv2
import numpy as np

# Create a Kalman filter object
kalman = cv2.KalmanFilter(4, 2)  

# Initialize state [[x, y, dx, dy], 
# where x, y is the location and dx, dy is the velocity
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

# Initialize other Kalman filter parameters
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
kalman.errorCovPost = np.eye(4, dtype=np.float32)

# Load the video
cap = cv2.VideoCapture('ball.avi')
predictions = []
while True:
    # Read each frame
    _, frame = cap.read()

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for the red ball
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red color
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=1)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    prediction = kalman.predict()
    prediction = (int(prediction[0][0]), int(prediction[1][0]))
    
     # Print the predicted position
    print(f"Predicted Position: {prediction}")
    
    predictions.append(prediction)
    #if there is more than one prediction, draw a line between them
    if len(predictions) > 1:
        for i in range(len(predictions)-1):
            cv2.line(frame,predictions[i],predictions[i+1],(0,255,0),2)

    # Find the largest contour, which is assumed to be the ball
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)

        # Compute the center of the contour
        M = cv2.moments(max_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        #print the actual position
        print(f"Actual Position: ", (cX, cY))
        
        #print the kalman prediction and the actual measurement
        print("Kalman Prediction: ", prediction)
        print("Measurement: ", (cX, cY))

        # Add measurement to the Kalman filter
        measurement = np.array([[cX], [cY]], dtype=np.float32)
        kalman.correct(measurement)
        
        # Predict the next position of the ball
        # prediction = kalman.predict()

        # Draw the predicted position on the frame
        
        # cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 5, (0,255,0), -1)
        # prediction = (int(prediction[0][0]), int(prediction[1][0]))
        # predictions.append(prediction)
        
        #Draw a line to show the trajectory of the ball.
        # cv2.line(frame,(int(prediction[0]), int(prediction[1])),(cX,cY),(255,0,0),5)
        
        #Draw a line to show the kalman filter prediction.
        # if len(predictions) > 1:
        #     for i in range(len(predictions)-1):
        #         cv2.line(frame,predictions[i],predictions[i+1],(255,0,0),5)
        # cv2.line(frame,(int(prediction[0]), int(prediction[1])),(int(prediction[0]+prediction[2]),int(prediction[1]+prediction[3])),(0,0,255),5)
        
        #print the kalman prediction and the actual measurement
        # print("Kalman Prediction: ", prediction)
        # print("Measurement: ", measurement.T)

        # Display the frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

