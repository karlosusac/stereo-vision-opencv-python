import cv2
import numpy as np
from matplotlib import pyplot as plt

#Function to create point cloud file
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

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
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num = len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

#Function that Downsamples image x number (reduce_factor) of times.
def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize = (col//2, row // 2))
	return image

#Load camera parameters
cameraParams = "camera_params\\"

ret = np.load(cameraParams + 'ret.npy')
K = np.load(cameraParams + 'K.npy')
dist = np.load(cameraParams + 'dist.npy')

#Load images
img_1 = cv2.imread("tank2L.jpg")
img_2 = cv2.imread("tank2R.jpg")

#Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size and height.
h,w = img_2.shape[:2]

#Get optimal camera matrix for better undistortion
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

#Undistort images
img_1_undistorted = cv2.undistort(img_1, K, dist, None, new_camera_matrix)
img_2_undistorted = cv2.undistort(img_2, K, dist, None, new_camera_matrix)

#Downsample each image 3 times (because they're too big)
img_1_downsampled = downsample_image(img_1_undistorted, 4)
img_2_downsampled = downsample_image(img_2_undistorted, 4)

#Set disparity parameters
#Note: disparity range is tuned according to specific parameters obtained through trial and error.
win_size = 5
min_disp = -1
max_disp = 63 #min_disp * 9
num_disp = max_disp - min_disp #Needs to be divisible by 16

#Create Block matching object.
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
	numDisparities = num_disp,
	blockSize = 5,
	uniquenessRatio = 5,
	speckleWindowSize = 5,
	speckleRange = 5,
	disp12MaxDiff = 2,
	P1 = 8*3*win_size**2, #8*3*win_size**2,
	P2 = 32*3*win_size**2) #32*3*win_size**2)

#Compute disparity map
print ("\nComputing the disparity  map...")
disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)

#Show disparity map before generating 3D cloud to verify that point cloud will be usable.
plt.imshow(disparity_map,'gray')
plt.show()

#Generate  point cloud.
print ("\nGenerating the 3D map...")

#Get new downsampled width and height
h,w = img_2_downsampled.shape[:2]

#Load focal length.
#focal_length = np.load('C:\\Users\\Stipe\\PycharmProjects\\stereo-vision-opencv-python\\Huawei-Y6-2019\\reconstruction\\camera_params\\FocalLength.npy')

'''
#Perspective transformation matrix
#This transformation matrix is from the openCV documentation, didn't seem to work for me.
Q = np.float32([[1,0,0,-w/2.0],
				[0,-1,0,h/2.0],
				[0,0,0,-focal_length],
				[0,0,1,0]])
'''
#This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision.
#Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf

Q2 = np.float32([[1,0,0,0],
				[0,-1,0,0],
				[0,0,0.35,0], #Focal length for Huawei Y6 2019 (26mm [35mm - film]).
				[0,0,0,1]])

#Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
#Get color points
colors = cv2.cvtColor(img_1_downsampled, cv2.COLOR_BGR2RGB)

#Get rid of points with value 0 (i.e no depth)
mask_map = disparity_map > disparity_map.min()

#Mask colors and points.
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

#Define name for output file
output_file = 'reconstructed.ply'

#Generate point cloud
print ("\n Creating the output file... \n")
create_output(output_points, output_colors, output_file)

