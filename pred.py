camera = cv2.VideoCapture(0) #uses webcam for video

while camera.isOpened():
    #ret returns True if camera is running, frame grabs each frame of the video feed
    ret, frame = camera.read()
    
    k = cv2.waitKey(10)
    if k == 32: # if spacebar pressed
        frame = np.stack((frame,)*3, axis=-1)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.reshape(1, 224, 224, 3)
        prediction, score = predict_image(frame)