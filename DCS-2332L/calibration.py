import cv2
import numpy as np
import glob
import PIL.ExifTags
import PIL.Image

#Define size of chessboard target.
# https://docs.opencv.org/2.4/_downloads/pattern.png
chessboard_size = (9, 6)

#Define arrays to save detected points
obj_points = []
img_points = []

#Prepare grid and points to display
objp = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)

objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

#https://eu.mydlink.com/download
video1 = cv2.VideoCapture('rtsp://admin:switch123@10.10.10.207/live1.sdp')

while (True):
    ret, frame1 = video1.read()
    cv2.imshow('camera1', frame1)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        gray_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        print("Image loaded, Analizying...")

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
        if ret == True:
            print("Chessboard detected!")
            # define criteria for subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # refine corner location (to subpixel accuracy) based on criteria.
            cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Calibrate camera
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_image.shape[::-1], None,
                                                         None)  # Save parameters into numpy file
        print('Saving!')
        np.save("./camera_params/ret", ret)
        np.save("./camera_params/K", K)
        np.save("./camera_params/dist", dist)
        np.save("./camera_params/rvecs", rvecs)
        np.save("./camera_params/tvecs", tvecs)
        print('saved!')
        break

video1.release()
cv2.destroyAllWindows()
