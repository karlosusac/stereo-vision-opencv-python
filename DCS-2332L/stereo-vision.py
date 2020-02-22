import cv2
import numpy as np

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


#https://eu.mydlink.com/download
video1 = cv2.VideoCapture('rtsp://admin:switch123@10.10.10.207/live1.sdp')
video2 = cv2.VideoCapture('rtsp://admin:switch123@10.10.10.209/live1.sdp')

while (True):
    ret, frame1 = video1.read()
    ret, frame2 = video2.read()

    cv2.imshow('kamera1', frame1)
    cv2.imshow('kamera2', frame2)

    imgL = cv2.pyrDown(frame1)
    imgR = cv2.pyrDown(frame2)

    window_size = 3
    min_disp = 16
    num_disp = 192 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                            numDisparities = num_disp,
                            uniquenessRatio = 10,
                            speckleWindowSize = 0,
                            speckleRange = 32,
                            disp12MaxDiff = 1,
                            P1 = 8 * 3 * window_size ** 2,
                            P2 = 32 * 3 * window_size ** 2
                            )
    disp = stereo.compute(imgL, imgR)
    cv2.imshow("disp", disp)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        h, w = imgL.shape[:2]
        f = 0.345 * w  #focal length for D-link DCS-2332L = 3.45 mm
        Q = np.float32([[1, 0, 0, -0.5 * w],
                        [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                        [0, 0, 0, -f],  # so that y-axis looks up
                        [0, 0, 1, 0]])
        points = cv2.reprojectImageTo3D(disp, Q)
        colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        mask = disp > disp.min()
        out_points = points[mask]
        out_colors = colors[mask]
        out_fn = 'out.ply'
        write_ply('out.ply', out_points, out_colors)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video1.release()
video2.release()
cv2.destroyAllWindows()
