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
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_VehicleDetectioAndTracking/blob/master/images/hog.png)
####3. Classifier training

The SVM classifier was the classifier choosen. Different type of classifier were tested. Different values for C, gamma parameters as well as different values for the kernel. 

Once adjusted the parameters, the most interesting points come from the kernel. Using a linear rather then an rbf kernel, I reached the following accuracies: 
1. 99.5% using the rbf kernel
2. 99.0% using the linear kernel. 

Some abservation about the two classifer are at the end of the report.

``` python
def train_SVC(self):

	if os.path.isfile(SVC_file):
		print("loading classifier from File...")
		data = pickle.load(open(SVC_file,"rb"))
		self.svc = data["svc"]
		self.std = data["std"]
	else:

		train_set, train_set_labels, test_set, test_set_labels = self.compute_dataset()

		print("Training SVC")
		if self.svc_type == "non-linear":
			self.svc = SVC(kernel = str(self.kernel)) #SVC(kernel='linear')
		if self.svc_type == "linear":
			self.svc = LinearSVC() #SVC(kernel='linear')
		self.svc.fit(train_set.tolist(), train_set_labels)
		score = self.svc.score(test_set,test_set_labels)
		predictions = self.svc.predict(test_set)
		print(score)
		print(predictions)
		data = {"svc":self.svc,"std":self.std}

		pickle.dump(data, open(SVC_file,"wb"))
```
The classifier was computed only once (as well as the the StarndardScaler)

```python
self.std = StandardScaler()

train_set = self.std.fit_transform(train_features)
test_set = self.std.fit_transform(test_features)
```

###Car detection and tracking

Once the classifier was trained, all the images from the frame were elaborated in order to detect and track the cars.

####1. Split of the frame into windows. 

The classifier was trained using windows of size (64,64). For this reason in order to detect cars into the frames of the video a windows split of the image is required. 
With the knowledge that cars appears bigger when close to the camera, and smaller when far away from the camera, in order to detect cars we need to use windows with different size, depending on the relative position respect to the camera. 
In addition, there is no reason to look for cars in the sky. So different region of interests where to look for cars were selected.

6 different region of interest were selected:
```python
search_windows = np.array([
	(0, 1280, 380, 600, 1, 2),
	(0, 1280, 380, 650, 1.5, 2),
	(0, 1300, 380, 700, 2, 1.5),
	(0, 1300, 380, 720, 2.5, 1.5),
	(0, 1300, 380, 720, 3, 1.25),
	(0, 1300, 380, 720, 3.5, 1.25)
])
```

Looking at the region of interests defined above, we can see overlap between the regions. The reason is because, for each of the reagion is also specified the scale factor and the overlap factor used during the sliding window procedure. Eeach item in the *search_windows* array above represent: *x_min, x_max, y_min, y_max, scale, overlap*. 
The most interesting elements in the entry of the array are probably the scale and the overlap parameters. While the overlap parameter is used in the sliding window function to specify how much to overlap each window to the previous one, the scale factor is used to scale the entire region of interest in order to apply always the same windows size. Means there is no a different size for the selected windwos, but instead the region of interest is scaled by a factor *scale*

```python
def compute_sliding_windows(self, image, x_start_stop=[None, None], y_start_stop=[None, None], scale = 1, cells_per_step = 2):

	roi = image[y_start_stop[0]:y_start_stop[1],x_start_stop[0]:x_start_stop[1],:]
	img_tosearch = cv2.resize(roi,(np.int(roi.shape[1]/scale),np.int(roi.shape[0]/scale)))
	if scale != 1:
		road_roi = RoadImage(img_tosearch)
	else:
		road_roi = RoadImage(roi)
	road_roi.extract_hog_features(self.hog_cspace, self.hog_channel, self.orient, 
								self.pix_per_cell, self.cell_per_block, False, False)

	nxbloks = (img_tosearch.shape[1] // self.pix_per_cell) -1
	nybloks = (img_tosearch.shape[0] // self.pix_per_cell) -1

	nblocks_per_windows = (self.windows_size //self.pix_per_cell) -1
	nxsteps = int((nxbloks - nblocks_per_windows) // cells_per_step)
	nysteps = int((nybloks - nblocks_per_windows) // cells_per_step)
	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = int(yb*cells_per_step)
			xpos = int(xb*cells_per_step)
	
			# Extract HOG for this patch
			hog_features = []
			hog_features.append(road_roi.hog_features_1[ypos:ypos+nblocks_per_windows, xpos:xpos+nblocks_per_windows]) 
			hog_features.append(road_roi.hog_features_2[ypos:ypos+nblocks_per_windows, xpos:xpos+nblocks_per_windows])
			hog_features.append(road_roi.hog_features_3[ypos:ypos+nblocks_per_windows, xpos:xpos+nblocks_per_windows])
			hog_features =  np.ravel(hog_features)

			x_left = int(xpos*pix_per_cell)
			y_top = int(ypos*pix_per_cell)
			# Extract the image patch

			subimg = cv2.resize(road_roi.image[y_top:y_top+self.windows_size, x_left:x_left+self.windows_size], (64,64))

			wi = WindowImage(subimg,hog_features)

			wi = self.compute_feature_vector(wi = wi)
			
			# Scale features and make a prediction
			test_features = self.std.transform(wi.image_features)

			test_prediction = self.svc.predict(test_features)

			if test_prediction == 1:
				xbox_left = np.int(x_left*scale)
				ytop_draw = np.int(y_top*scale)
				win_draw = np.int(self.windows_size*scale)
				self.windows.append(((xbox_left+x_start_stop[0], ytop_draw+y_start_stop[0]),(xbox_left+win_draw++x_start_stop[0],ytop_draw+win_draw+y_start_stop[0])))
```
The first two code lines select the region of interest and then resize it by the scale factor

```python
roi = image[y_start_stop[0]:y_start_stop[1],x_start_stop[0]:x_start_stop[1],:]
img_tosearch = cv2.resize(roi,(np.int(roi.shape[1]/scale),np.int(roi.shape[0]/scale)))
```
Then all the parameters of roi splitting are computed and used to select window image, to be converted into feature vector and then classified.
```python

wi = WindowImage(subimg,hog_features)

wi = self.compute_feature_vector(wi = wi)

# Scale features and make a prediction
test_features = self.std.transform(wi.image_features)

test_prediction = self.svc.predict(test_features)
```

Finally, if the window is classified as car, this is addedd to the windows array for the drawing step.

####2. Heatmap computation

In order to track cars over some frames (I choose 5 as number of frames over which to track cars), and mainly to remove false positive, the heatmap of each detection was computed. 
For each frame computed, the class *DetectionManager* keeps track of all the windows detected in the *self.windows* list. 
All the windows are firstly computed in order to add them in the list of the windows of the last frame seen over the last 5. Then the heatmap is computed for this single frame and combined with the last 4 by the DetectionManager

```python	
def manage_heat_maps(self, image):
	heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
	# Add += 1 for all pixels inside each bbox
	# Assuming each "box" takes the form ((x1, y1), (x2, y2))
	for box in self.windows:
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	if len(self.heat_maps) >= self.heat_map_frames:
		del(self.heat_maps[0])
		del(self.windows_history[0])

	self.heat_maps.append(heatmap)
	self.windows_history.append(self.windows)
	self.apply_threshold()

def filter_window_by_heatmaps(self,image):

	labels = label(self.heatmap)
	return self.draw_labeled_bboxes(np.copy(image), labels)
```
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_VehicleDetectioAndTracking/blob/master/images/heatmap.png)

####3. Car windows drawing

Finally, the windows filtered by heatmaps are drawed in the original image.

```python	
def draw_labeled_bboxes(self, img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	# Return the image
	return img
```
![alt tag](https://github.com/ciabo14/SelfDrivingCarND_VehicleDetectioAndTracking/blob/master/images/car_detected.png)
###Pipeline 

The execution of the described pipeline to a video respect to an image has 1 main difference: the application of the thracking respect to the last x frames. 
Even if the heatmap can be computed as well and the false positive can be removed, cannot be applied the detection over all the last frames.
In both cases the classifier is trained, or the already trained classifier is loaded from the pickle file.

For video pipeline, the DetectionManager is defined only once and the used for all the frames, in order to use the history about detection.
The *moviepy.editor VideoFileClip* class was used to elaborate the video. 

```python
def test_on_videos(dm):
    files = os.listdir('./Dataset/videos/')
    for f in files:
        test_on_video(dm,"./Dataset/videos/",f)

def process_image(img):

    result = dm.detect_cars(img)
    return result
    
if __name__ == "__main__":
    
    dm = DetectionManager()
    dm.train_SVC()
    test_on_video(dm,"./Dataset/","project_video.mp4")
   
```

###Discussion

####1. 

At the end of the project there are some points that interest me during the development:
1. Dataset management. I used for this project the provided dataset that comes from some video stream. Means that only using a simple random selector of images for train and test set may have brought into a "not balanced" dataset, in the sense that some train examples could not have been used also for validation (and viceversa), reducing the generalization of the classifier. Moreover we use a bunch of examples for this project (almost 20000). We know that SVM does not work properly with very big dataset. But which is the max value?
2. SVM Classifier kernel. During the development of this project I tryed both the linear and the rbf kernels. While the second one brings to better results during the training phase, the first one seems to work better in the video. The classifier with rbf kernel as a matter of facts, had an hard work in classifying correclty cars close to the camera, expecially for the black one. On the other side it introduce almost none false positive. My feeling in this case is that the non linear svm system overfit data and does not generalize weel.  

####2 
Interesting possibile future investigation
Several are the possibile interesting investigation:

1. Try to find different features that can represent even better a car. For the purpose of this project we use color, edges/shapes, and spatial features. Perhaps something different and powerful can be thought
2. Try with different classifier as well as different kernel for the svm classifier.
3. Why not combine the computer vision approach with a strong DNN?
