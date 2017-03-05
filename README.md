# Project 5: Vehicle detection and tracking
This is the 5th project for the Self driving car nanodegree - Term 1. This requires to elaborate the images coming from a camera placed in front of the car, in order to detect cars and track them.

The provided project is made of the following files:
[//]: # (File References)
[***main.py***]: is the files containing the main, that runs the entire algorithm in the test images rather then in the videos

[***RoadImage.py***]: is the basic class which represent a road image. It includes some image representation and elaboration used during the computation

[***WindowImage.py***]: is the basic class which represent a window. It is used both in the treaning phase and in the classification step. It includes all the logic for feature extraction. 

[***DatasetManager.py***]: the DatasetManager is the manager class used to elaborate the dataset. It includes the train/test split as well as the vehicle/non-vehicle management

[***DetectionManager.py***]: this is the core class of the project. It includes all the training computation for the classifier as well as the second part involved in the classification and vehicle detection and traction. 

[***output images***]: ./output_images are some images coming from the video stream and elaborated with the sw.

[***output video***]: ./project_video_processed.mp4 is the project_video.mp4 reprocessed accordingly to the sw computed.

---

###Classifier training

In order to detect correctly the cars in each frame of the video, some kind of classifier need to be trained with pictures form the provided dataset. 

####1. Dataset management

The dataset provided is made as follows:
vehicles: 2876 + 7100
non-vehicles: 8968

The vehicles dataset was made of two subset. The first includes usual vehicle images. The second set instead, comes from vehicles images of different nature, in more critical scenario. 
Below a couple of images coming from the datasets.

![alt tag](https://github.com/ciabo14/SelfDrivingCarND_VehicleDetectioAndTracking/blob/master/images/non-vehicles.png)
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_VehicleDetectioAndTracking/blob/master/images/vehicles1.png)
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_VehicleDetectioAndTracking/blob/master/images/vehicles2.png)

In order to let the classifier be as much as generalized as possible, both the two dataset were used.
All the images are firstly detected by the datasetManager, and splitted into the train/test sets with a 80%/20% ratio.
The split was done using the train_test_split function provided by *sklearn* library:

```python
for (path, dirs, f) in os.walk(dataset_positive):
            np.random.shuffle(f)
            files1 = ["{}/{}".format(dataset_positive,file_name) for file_name in f]
            labels1 = [1 for i in files1]
        for (path, dirs, f) in os.walk(dataset_negative):
            np.random.shuffle(f)
            files2 = ["{}/{}".format(dataset_negative,file_name) for file_name in f]
            labels2 = [0 for i in files2]
        
        X_train, X_test, y_train, y_test = train_test_split(files1+files2, labels1+labels2, test_size=test_p, random_state=42)
```
####2. Feature extraction

In order to train the classifier, features for each image were computed. Accordingly with the class videos, three different type of features were computed for each image:

1. spatial_features;
2. histogram_features;
3. hog_features

The three different type of features represent three different aspects of images and together well represent each image.
While spatial and histogram features are striclty correlated with the color of pixels, the hog feature is correlated with the shape of the image. For this reason the first two feature were computed in the RGB representation of the image, while the hog features were computed from the HLS representation of the image itself.

```python
self.spatial_cspace = "RGB"
		self.hist_cspace = "RGB"
		self.hog_cspace = "HLS"

...

def compute_feature_vector(self,image_name = "",wi = None):
		
		if image_name != "":
			image = WindowImage(cv2.cvtColor(cv2.imread(image_name),cv2.COLOR_BGR2RGB))
		if wi != NullType:
			image = wi
		if self.spatial_f_en:
			image.extract_spatial_features(self.spatial_cspace, self.spatial_size)
		if self.color_hist_f_en:
			image.extract_hist_features(self.hist_cspace, self.hist_bins, self.hist_range)
		if self.hog_f_en and wi == None:
			image.extract_hog_features(self.hog_cspace, self.hog_channel, self.orient, self.pix_per_cell, self.cell_per_block, False, False)
		
		image.combine_features(self.spatial_f_en, self.color_hist_f_en, self.hog_f_en)
		return image
```
The *compute_feature_vector* is executed for each image in the dataset, and exploit the feature computation defined directly in the *WindowImage* class. 
Note that the second condition for the computation of the *hog_feature* vector is used to avoid the computation of the hog features multiple time in the classification phase.
After several tests, the following values where choosen as parameters for the feature extraction:
``` python
self.orient = 9
self.pix_per_cell = 8
self.cell_per_block = 2 
self.hog_channel = "ALL"
self.spatial_size = 16
self.hist_bins = 32
self.hist_range = (0,256)
```

####3. Classifier training

The SVM classifier was the classifier choosen. Several 

###Pipeline (single images)

Once the camera is calibrated (ad this is executed only once whe I started working on the project), each image (single or from a frame), can be elaborated in order to detect lane.
As the graph below shows, the images are processed as follow:
1. Undistort the image using the camera matrix and the undistortion parameters as shown above
2. Apply image analisys *filters* in order to keep almost only pixels from lane lines:
  1. Apply HLS image thresholds over Saturation and Hue channels
  2. Apply sobel operator along x and y direction; 
    1. Combine sobel x and sobel y masks with magnitude thresholds
    2. Combine sobel x and sobel y masks with direction thresholds
    3. Combine the both filters above in a Bitwise OR manner
3. Combine filter both from Sobel application and color channels thresholding in a Bitwise OR manner.
4. Apply perspective transformation to select lane section only
5. Elaborate the image obtained from the perspective transformation application as follow:
  1. Compute lane lines pixels using slinding windows and the histogram, or using the last recognised lane lines
  2. Evaluate the found lines from a plausability point of view
  3. Compute lines polynomial fit
  4. Compute lanes curvature and position
6. Transform the binary mask with the found lane lines back in the original perspective 
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/Pipeline_Diagram.png)  

####1. Correction of the image distortion.

Test images, or frames from the video, were undistorted using the CameraManager function *undistort_image()*. This function only take an image as input, and return the undistorted image 
```python
def undistort_image(self,img):
	return cv2.undistort(img, self.cam_mtx, self.cal_dist, None, self.cam_mtx)
```
Below an example of how an image appears after the distortion correction.
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/TestImageUndistortion.png)  

####2. Image filtering

Lines are elements in the image recognized by drivers because of their shape, color and position/direction. Moreover, lines are detected in different light conditions. Good Light, presence of shadows ecc. 
In order to recognize lines as a human driver does, the same recognition process is eligible for machines. 
For this reasons two different filtering to the images were applied in order to discover lane lines:
1. Color filtering
2. Shape and position filtering using Sobel operator
The first intuitive way to use color filtering, is to filter white and yellow colors in the image and discard all the other colors. However, using the RGB color space, we can develop a filter correlated to the light in the image (enviroment). 
However, moving to a different space we can capture lines indipendetly form the the light (day light, artificial lights or even shadows). 
This is the case of the HLS color space. 
Filtering the image in the Hue and Saturation channels, we are able to remove the majority of the pixels keeping lane lines even in shadows conditions.
I applyed a threshold to the Hue and Saturation channels defined as below:
```python
mask_HLS = {"low_thresholds":np.array([ 0,  0,  100]), "high_thresholds":np.array([ 100, 255, 255])}
...

def color_select(self, mask_dict, img_representation = "RGB"):
	mask = np.zeros_like(self.gray)
	if(img_representation == "RGB"):
		mask[((self.undistorted_image[:,:,0] >= mask_dict["low_thresholds"][0]) & (self.undistorted_image[:,:,0] <= mask_dict["high_thresholds"][0])) & ((self.undistorted_image[:,:,1] >= mask_dict["low_thresholds"][1]) & (self.undistorted_image[:,:,1] <= mask_dict["high_thresholds"][1]))& 
	
	if(img_representation == "HLS"):
		mask[((self.HLS_image[:,:,0] >= mask_dict["low_thresholds"][0]) & (self.HLS_image[:,:,0] <= mask_dict["high_thresholds"][0]))&((self.HLS_image[:,:,1] >= mask_dict["low_thresholds"][1]) & (self.HLS_image[:,:,1] <= mask_dict["high_thresholds"][1]))&((self.HLS_image[:,:,2] >= mask_dict["low_thresholds"][2]) & (self.HLS_image[:,:,2] <= mask_dict["high_thresholds"][2]))] = 1	

	return mask
```
Below you can find the application of the HLS color filtering to one of the test images.
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/HLSFIltering.png)  
**Sobel operator** is a very powerful operator to detect edges. Depending on the kernel size, it can detect sharper or stronger edges in the desired direction. Edges are computed convolving the kernel all along the image and computing a gradient value for each pixel of the image. Thresholding this gradient let us to choose which pixels are for us edges (an then lane lines).
Sobel application along x and y, can be combined in different ways. For this project:
1. I first combined the two direction application looking at the magnitude mask
2. Then I used the x and y application of the operator to compute a sobel direction mask
3. I combined the magnitude and the direction mask in a single sobel mask.  
The ImageManager is responsible to aply this filtering with the function *combine_sobel_filters()*. It first calls the RoadImage *apply_sobel_operator()* that runs the procedure of filters computation
```python
self.mag_thresh = (30,255)
self.sobel_kernel = 5
self.dir_thresh = (0.6, 1.2)

...

def apply_sobel_operator(self):

	sobel_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
	sobel_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
	self.abs_sobel_x = self.abs_sobel_thresh(sobel_x,"x")
	self.abs_sobel_y = self.abs_sobel_thresh(sobel_y,"y")
	self.mag_sobel = self.mag_thresh_mask(sobel_x,sobel_y)
	self.dir_sobel = self.dir_threshold(sobel_x,sobel_y)
```
and then combine the sobel filters application as:
```python
def combine_sobel_filter(self,image):
	sobel_combined = np.zeros_like(image.gray)
	#sobel_combined[((image.abs_sobel_x == 1) & (image.abs_sobel_y == 1)) | ((image.mag_sobel == 1) & (image.dir_sobel == 1))] = 1
	sobel_combined[((image.mag_sobel == 1) & (image.dir_sobel == 1))] = 1
	return sobel_combined
```
This bring in results like in the image below:
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/SobelFiltering.png)  

####3. Color and Sobel masks combination

Finally, color and sobel masks are combined  in a Bitwise OR manner, leading at the following edge image:
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/HLS_SobelMasksApplication
.png)
Below The code that describe all the filtering process executed by the ImageManager class:
```python
def filter_image_for_line_detection(self):
	self.img.apply_sobel_operator()
	self.img.set_sobel_combined(self.combine_sobel_filter(self.img))
	self.img.set_color_line_mask(self.combine_color_filters(self.img))
	self.img.set_lane_lines_mask(cv2.bitwise_or(self.img.sobel_combined,self.img.color_line_mask))
```

####4. Perspective transformation

Perspective transformation to bird eyes perspective is very useful to limit the section of the image where to focus the interest and, more important, to work on an image without prospective distortion (parallel lines appears parallel in the bird eyes image and not convergent in the vanishing Point).
For this purpose the *cv2.warpPerspective* was computed it the edge image, selecting as source and destination corners of the rectangle the following corners:
Below The code that describe all the filtering process executed by the ImageManager class:
```python
height_section = np.uint(img_size[1]/2)

top_left_coordinate = height_section - .107*np.uint(img_size[1]/2)
top_right_coordinate = height_section + .113*np.uint(img_size[1]/2)
bottom_left_coordinate = height_section - .7*np.uint(img_size[1]/2)
bottom_right_coordinate = height_section + .75*np.uint(img_size[1]/2)

top_margin = np.uint(img_size[0]/1.55)
bottom_margin = np.uint(img_size[0])

src_corners = np.float32([[bottom_left_coordinate,bottom_margin], #bottomLeft
	[bottom_right_coordinate,bottom_margin],	#bottomRight
    [top_right_coordinate,top_margin], #topRight
    [top_left_coordinate,top_margin]]) #topLeft
    
"""
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 192, 720      | 200, 720      | 
| 1120, 720     | 1080, 720     |
| 712, 464      | 1080, 0       |
| 571, 464      | 200, 0        |
 """    
```
I also tryed with different fixed corners. However this bring very similar results.
```python
src_corners = np.array([[585, 460], [203, 720], [1127, 720], [695, 460]]).astype(np.float32)
			dst_corners = np.array([[320, 0], [320, 720], [960, 720], [960, 0]]).astype(np.float32)
```
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/PerspectiveTransformation.png)
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_AdvancedLaneFinding/blob/master/images/PerspectiveTransformation-Filtered.png)

####5. Lane lines detection

Since having a significative image on which working on as basilar for lane detection, a roboust approach to detect lane lines is as much important as a roboust filtering of the original image.
In order to achieve the goal to detect lane lines (and than the entire lane), some consecuteve steps where executed:
1. Compute lane lines pixels using slinding windows and the histogram, or using the last recognised lane lines
2. Evaluate the found lines from a plausability point of view
3. Compute lines polynomial fit
4. Compute lanes curvature and position
This sequence of computation are executed by the function *detect_lines()* of the Line class
def detect_lanes(self, binary_warped):
```python
def detect_lanes(self, binary_warped):
	left_pixel_positions_x,left_pixel_positions_y,right_pixel_positions_x,right_pixel_positions_y = self.pixels_detection(binary_warped)
	self.manage_detected_pixels(left_pixel_positions_x, left_pixel_positions_y, right_pixel_positions_x, right_pixel_positions_y)
	self.fit_found_lanes(binary_warped)
	self.manage_curvature()

	self.find_offset()
```

#####5.1 Compute lane lines position pixels *pixels_detection()*

Depending on the application (test images or video stream), and from the history of detection (in case of video stream), the algorithm detect the lane pixels: 
1. or appliying a sliding windows starting from the peaks found in the histogram of the image, 
2. or looking at the pixels around the last lines detected.
The sliding windows approach is applied when we are using a single test image; when we are looking at the first frame of a video or when the left or right lines are not detected for the last *x* frames:
```python
# In case first frame or in last 5 frames I did not found a left or right line
	if(self.first_frame or self.left_line_missing > 5 or self.right_line_missing > 5):
```
In all the other cases, pixels around the last lines are considered for the new line detection.
```python
def pixels_detection(self,binary_warped):
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	left_lane_inds = []
	right_lane_inds = []

	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
...
```

#####5.2 Evlaluate found lines *manage_detected_pixels()*

Once candidate lines pixels were selected, the strategy need to decide if the found pixels are enough for considering a lane line, and if these pixels brings to a significative lane line. 
Two different approaches were tested: the first one use a buffer with the last *self.last_frame_used* lines detected pixels, and depending on the amount of the pixels detected, the list ring is modified accordingly. In case enough pixels were detected (and then we can consider the line as detected), this amout of pixels were added to the list while the oldest detected pixels in the list were removed: if the amount of pixels are not eought, the last detected pixels in the list were added once more to the list itself, removign the oldest pixels detected.
The second approach used instead, I consider the detected pixels as a line and, indipendently from the number of pixels detected, these are used for polinomial fitting. 
```python
def manage_detected_pixels(self, left_x, left_y, right_x, right_y):

	self.left_x = left_x
	self.left_y = left_y
	self.right_x = right_x
	self.right_y = right_y

	if self.use_lines_history:
		num_frames = len(self.last_frames_left_x)

		if(num_frames == 0):
			self.last_frames_left_x.append(left_x)
			self.last_frames_left_y.append(left_y)
			self.last_frames_right_x.append(right_x)
			self.last_frames_right_y.append(right_y)				
		else:
			if num_frames >= self.last_frame_used:
				del(self.last_frames_left_x[0])
				del(self.last_frames_left_y[0])
				del(self.last_frames_right_x[0])
				del(self.last_frames_right_y[0])

			if len(left_x) > self.min_pix_line_identification:
				self.last_frames_left_x.append(left_x)
				self.last_frames_left_y.append(left_y)
				self.left_line_missing = 0
			else:
				self.last_frames_left_x.append(self.last_left_x)
				self.last_frames_left_y.append(self.last_left_y)
				self.left_line_missing += 1
			if len(right_x) > self.min_pix_line_identification:
				self.last_frames_right_x.append(right_x)
				self.last_frames_right_y.append(right_y)
				self.right_line_missing = 0
			else:
				self.last_frames_right_x.append(self.last_right_x)
				self.last_frames_right_y.append(self.last_right_y)
				self.right_line_missing += 1
	else:
		if self.last_left_x != None:
			if len(left_x) < self.min_pix_line_identification:
					self.left_x = self.last_left_x
					self.left_y = self.last_left_y
			if len(right_x) < self.min_pix_line_identification:
					self.right_x = self.last_right_x
					self.right_y = self.last_right_y
		
	self.set_last_xy(left_x, left_y, right_x, right_y)
```

#####5.3 Compute lines polynomial fit *fit_found_lanes()*

In both cases the pixels are then fitted by a polynomial function. In the first case all the detected pixels from history are used to fit a polynomial function. In second case instead, the polinomial function fitted in the last detected pixels is weighted with the polynomial coefficients found at the last iteration.
def fit_found_lanes(self, binary_warped):
```python
def fit_found_lanes(self, binary_warped):

	# Fit a second order polynomial to each
	if self.use_lines_history:
		current_left_x = [item for sublist in self.last_frames_left_x for item in sublist]
		current_left_y = [item for sublist in self.last_frames_left_y for item in sublist]
		current_right_x = [item for sublist in self.last_frames_right_x for item in sublist]
		current_right_y = [item for sublist in self.last_frames_right_y for item in sublist]
	else:
		current_left_x = self.left_x
		current_left_y = self.left_y
		current_right_x = self.right_x
		current_right_y = self.right_y

	self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

	l_fit = np.polyfit(current_left_y, current_left_x, 2)
	r_fit = np.polyfit(current_right_y, current_right_x, 2)

	if(self.use_lines_history):
		self.left_fit = l_fit
		self.right_fit = r_fit
	else:
		self.compute_smoothed_poly(l_fit, r_fit)

	self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
	self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
```

#####5.4 Compute line curvature and camera position *manage_curvature(),  find_offset()*

Lane curvature can be an important indicator about the the detected lane lines are correct, and even more importantly, can be an important indicator about how to handle a courve with a steering angle.
For this reason, starting from detected lines, the curvature of these is computed. As for the lines pixels, also for the curvature two different approaches were tested: the first who involves the last *self.last_frame_used* curvature value to smooth the current one; the second one who smooth the current computed curvature, the the curvature computed in the last frame.
This is not all. Since curvature is highly dependant from the detected lines, and the lines from the pixel (i.e. from the edges detected in the first stage of the project), a plausability check was done, before to use this curvature value as value for the current frame.
```python
def manage_curvature(self):

	l_curvature = self.estimate_Rof("l")
	r_curvature = self.estimate_Rof("r")

	if self.use_lines_history:

		left_mean = np.mean(self.last_frame_left_curvature)
		right_mean = np.mean(self.last_frame_right_curvature)

		num_frames = len(self.last_frame_left_curvature)
		tmp_curvature = 0

		if num_frames == 0:
			self.last_frame_left_curvature.append(l_curvature)
			self.last_frame_right_curvature.append(r_curvature)
			self.mean_curvature = np.mean([l_curvature,r_curvature])
		else:
			if num_frames >= self.last_frame_used:
				del(self.last_frame_left_curvature[0])
				del(self.last_frame_right_curvature[0])

			if left_mean + self.max_curvature_deviation > l_curvature > left_mean - self.max_curvature_deviation:
				self.last_frame_left_curvature.append(l_curvature)
			else:
				self.last_frame_left_curvature.append(left_mean)
				l_curvature = left_mean

			if right_mean + self.max_curvature_deviation > r_curvature > right_mean - self.max_curvature_deviation:
				self.last_frame_right_curvature.append(r_curvature)
			else:
				self.last_frame_right_curvature.append(right_mean)
				r_curvature = right_mean

		self.mean_curvature = np.mean([l_curvature,r_curvature])

	else:
		self.compute_smoothed_curvature(l_curvature, r_curvature)

```
In the first approach, in case the computed curvature is not close enough to the last one (or the mean depending on the history curvature), it is discarded and the the mean of the curvature history is used as current curvature. 

The camera position finally is computed using the lines detected as described above, with the following function:
```python
def find_offset(self):
	lane_width = 3.7  # metres
	h = 720  # height of image (index of image bottom)
	w = 1280 # width of image

	# Find the bottom pixel of the lane lines
	l_px = self.left_fit[0] * h ** 2 + self.left_fit[1] * h + self.left_fit[2]
	r_px = self.right_fit[0] * h ** 2 + self.right_fit[1] * h + self.right_fit[2]

	# Find the number of pixels per real metre
	scale = lane_width / np.abs(l_px - r_px)

	# Find the midpoint
	midpoint = np.mean([l_px, r_px])

	# Find the offset from the centre of the frame, and then multiply by scale
	self.offset = (w/2 - midpoint) * scale
```

###6. Drow information on the original undistorted image

Finally, both for test images and for frames caming from a video stream, the information are drown in the bird eyes image, and than transformed back in the original unistorted image.

###Pipeline (video)

The execution of the described pipeline to a video respct to an image has 2 main differences:
1. All the detection smoothing as well as plausability verification can be applied
2. The frames need to be written back in a video stream.
The second requirement was accomplished with the support of the *moviepy.editor VideoFileClip* class.
Instead of just apply the algorithm to a single image (and create a new ImageManager for each image), a single instance of the Image manager is used and history about lanes and images is memorized

```python
def test_video():
	print("Running on test video1...")
	# Define our Lanes object
	#im = ImageManager(cm)
	#####################################
	# Run our pipeline on the test video 
	#####################################

	clip = VideoFileClip("./project_video.mp4")
	output_video = "./project_video_processed.mp4"
	output_clip = clip.fl_image(process_image)
	output_clip.write_videofile(output_video, audio=False)

def process_image(img):
	result = im.find_lane_lines(img)
	return result
```

###Discussion

####1. 
In my opinion, the good results of a solution to this kind of problem comes from two different ways:
1. Compute a strong and roboust edge detection to identify lane lines
2. Develop a smart strategy to detect lines depending of the history (last frames) and the current detection features (like difference with the last detection rather than plausible curvature).

A good combination of the two point above can provide good lane detection in almost all the conditions. 
Of course, more complicated situations with a lot of shadows or artificial and not constant light, or even sun light reflection, requires a stronger calibration of the approach parameters.

####2 
Interesting possibile future investigation
Several are the possibile interesting investigation:

1. Apply all the strategy (from the edge detection using color and sobel operator) at the bird eye image instead of at the original image. This would let the approach to ignore from the start about all the not interesting points
2. Investigate in deep masks combinations for edge detection. Different operators as well as different combinatio of the computed masks could bring to different solution
3. Why not combine the computer vision approach with a strong DNN?
