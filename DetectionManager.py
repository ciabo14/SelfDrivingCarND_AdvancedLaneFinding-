from CarDetection.WindowsManager import WindowsManager
from CarDetection.DatasetManager import DatasetManager

import os.path
import numpy as np
import pickle
from CarDetection.RoadImage import RoadImage
from CarDetection.WindowImage import WindowImage
import cv2
import matplotlib.pyplot as plt
import progressbar
import time

from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from theano.gof.null_type import null_type
from theano.gof.fg import NullType

SVC_file = "classifier.p"
dataset = ""

# The *en parameters are used to enable or disable the computation and the usage of certain types of features
spatial_cspace = "RGB"
hist_cspace = "RGB"
hog_cspace = "HLS"
orient = 9
pix_per_cell = 8
cell_per_block = 2 
hog_channel = "ALL"
spatial_size = 16
hist_bins = 32
hist_range = (0,256)
spatial_f_en = True
color_hist_f_en= True
hog_f_en= True
windows_size = 64
# Define the number of frames over which to compute the heatmap
heat_map_frames = 5

class DetectionManager:

	def __init__(self):
		self.dataset_manager = DatasetManager()
		self.heat_maps = []

		self.spatial_cspace = "RGB"
		self.hist_cspace = "RGB"
		self.hog_cspace = "HLS"
		self.orient = 9
		self.pix_per_cell = 8
		self.cell_per_block = 2 
		self.hog_channel = "ALL"
		self.spatial_size = 16
		self.hist_bins = 32
		self.hist_range = (0,256)
		self.spatial_f_en = True
		self.color_hist_f_en= True
		self.hog_f_en= True
		self.windows_size = 64
		# Define the number of frames over which to compute the heatmap
		self.heat_map_frames = 5
		self.svc_type = "linear" 
		self.kernel = "rbf"
		
		self.windows = []
		self.windows_history = []
		self.heat_maps = []
		self.threshold = 15
	
	def detect_cars(self,image):
		
		self.detect_cars_via_slidingwindows(image)
		self.manage_heat_maps(image)
		return self.filter_window_by_heatmaps(image)
		
	def detect_cars_via_slidingwindows(self,image):
		
		# The search_windows array express the region of interest where to apply sliding windows to search for cars. Each tuple includes:
		# (x_min, x_max, y_min, y_max, scale_factor, overlap) where the first 4 are the top left and bottom right corners; the fifth is the scale factor to be applied to the image;
		# the sixth is overlap factor
		print("Detecting cars...")
		search_windows = np.array([
		
	    (400, 1280, 380, 600, 1, 2),
	    (400, 1280, 380, 650, 1.5, 2),
	    (400, 1300, 380, 700, 2, 1.5),
	    (400, 1300, 380, 720, 2.5, 1.5),
	    ##(0, 1280, 380, 720, 2.75, 1.5),
	    (400, 1300, 380, 720, 3, 1.25),
	    ##(0, 1300, 380, 720, 3.1),
	    ##(0, 1280, 380, 720, 3.2),
	    ##(0, 1280, 380, 720, 3.3),
	    ##(0, 1280, 380, 720, 3.4),
	    (400, 1300, 380, 720, 3.5, 1.25),
	    ##(0, 1280, 380, 720, 3.6),
	    ##(0, 1280, 380, 720, 3.7),
	    #(0, 1280, 380, 720, 3.75, 1.25),
	    ##(0, 1280, 380, 720, 3.8),
	    ##(0, 1280, 380, 720, 3.9)
	    #(0, 1300, 380, 720, 4, 1.15),
	    #(0, 1300, 380, 720, 4.25, 1.15),
	    #(0, 1300, 380, 720, 4.5, 1.15),
	    #(0, 1300, 380, 720, 4.75, 1.15),
	    #(0, 1300, 380, 720, 5, 1.15)
	    ##(900, 1280, 380, 720, 4.2),
	    ##(900, 1280, 380, 720, 4.4),
	    ##(900, 1280, 380, 720, 4.6),
	    ##(900, 1280, 380, 720, 4.8),
	    ##(900, 1280, 380, 720, 5)
	    ])
		self.windows = []

		for x_min, x_max, y_min, y_max, s, c in search_windows:
			self.compute_sliding_windows(image, x_start_stop=[int(x_min), int(x_max)], y_start_stop = [int(y_min), int(y_max)], scale = s, cells_per_step = c)		

		print("Detected {} windows".format(str(len(self.windows))))
	def draw_car_rectangles(self, image):
		
		draw_img = np.copy(image)
		for windows in self.windows_history:
			for window in windows:
				cv2.rectangle(draw_img,window[0],window[1],(0,0,255),6) 
		
		print("# of squares found:",len(self.windows))
		return draw_img
		
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
	
	def apply_threshold(self):

		self.heatmap = np.zeros_like(self.heat_maps[0]).astype(np.float)
		
		for heat in self.heat_maps:	
			self.heatmap += heat
		# Zero out pixels below the threshold
		self.heatmap[self.heatmap <= self.threshold] = 0
		# Return thresholded map
			
	#Function used to test different classifier with the provided datasets
	def test_SVC_training(self):
		cspaces = np.array(['RGB','HLS','HSV'])
		orients = np.array([9,15,19,25,32])
		spatial_sizes = np.array([16,32,48])
		hist_binss = np.array([64,128,256])
		C = np.array([10,1,.1,.01,.001])
		kernels = np.array(['linear', 'poly', 'rbf'])
		scores = []
		for spatial_cspace in cspaces:
			hist_cspace = spatial_cspace
			for hog_cspace in cspaces:
				for orient in orients:
					for spatial_size in spatial_sizes:
						for hist_bins in hist_binss:
							train_set, train_set_labels, test_set, test_set_labels = self.compute_dataset(spatial_cspace, 
								spatial_size, hist_cspace, hist_bins, hist_range, hog_cspace, hog_channel, orient, pix_per_cell, 
								cell_per_block)
							for c in C:
								for kernel in kernels:
									print('spatial_cspace={}-hog_cspace={}-orient={}-spatial_size={}-hist_bins={}-c={}-kernel={}'
										.format(spatial_cspace,hog_cspace,orient,spatial_size,hist_bins,c,kernel))
									self.svc = SVC(C=c,kernel = str(kernel)) #SVC(kernel='linear')
									self.svc.fit(train_set.tolist(), train_set_labels)
									score = self.svc.score(test_set,test_set_labels)
									predictions = self.svc.predict(test_set)
									print('score'+str(score))
									scores.append('spatial_cspace={}-hog_cspace={}-orient={}-spatial_size={}-hist_bins={}-c={}-kernel={}---score={}\n'
										.format(spatial_cspace,hog_cspace,orient,spatial_size,hist_bins,c,kernel,score))
		f = open('log.data', 'w')
		f.writelines(scores) 
		f.close()
								
	def train_SVC(self):
		# if the classifier was already trained, it is loaded from the file classifier.p
		if os.path.isfile(SVC_file):
			print("loading classifier from File...")
			data = pickle.load(open(SVC_file,"rb"))
			self.svc = data["svc"]
			self.std = data["std"]
		else:
			
			train_set, train_set_labels, test_set, test_set_labels = self.compute_dataset()

			print("Training SVC")

			#depending on the svc_type parameter a linear rather then non linear parameter is choosen
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
			
			
	def compute_dataset(self, spatial_cspace = spatial_cspace, spatial_size = spatial_size, hist_cspace = hist_cspace, 
								hist_bins = hist_bins, hist_range = hist_range, hog_cspace = hog_cspace, hog_channel = hog_channel, 
								orient = orient, pix_per_cell = pix_per_cell,cell_per_block = cell_per_block):
		
		print("Splitting dataset into train and test sets")
		train_set_images, train_set_labels, test_set_images, test_set_labels = self.dataset_manager.compute_dataset()
		
		print("Computing features for train and test sets examples")
		train_features, test_features = self.compute_features_for_train(train_set_images, test_set_images, spatial_cspace, 
								spatial_size, hist_cspace, hist_bins, hist_range, hog_cspace, hog_channel, orient, pix_per_cell, 
								cell_per_block)
		
		print("Normalizing features examples")
		self.std = StandardScaler()
			
		train_set = self.std.fit_transform(train_features)
		test_set = self.std.fit_transform(test_features)
		
		return train_set, train_set_labels, test_set, test_set_labels
	
	def compute_features_for_train(self, train_set_images, test_set_images, spatial_cspace, spatial_size, hist_cspace, 
								hist_bins, hist_range, hog_cspace, hog_channel, orient, pix_per_cell, cell_per_block):
		
		train_set = []
		test_set = []

		print("Computing Training set features")

		for image_name in train_set_images:
			image = self.compute_feature_vector(image_name = image_name)

			train_set.append(image.image_features)

		
		print("Computing Test set features")
		for image_name in test_set_images:
			image = self.compute_feature_vector(image_name = image_name)
			
			test_set.append(image.image_features)
		return train_set, test_set
	
	# This function compute a feature vector for an image with name image_name, or for a given image 
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
	
	# Compute the sliding windows depending on the roi (defined by x_start_stop and y_start_stop), the scale factor and the "overlap"
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
		draw_img = np.copy(image)
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
	
	# Print iterations progress
	def compute_features_for_single_image(self, img, q):
		
		wi = WindowImage(img)
		wi.extract_spatial_features(self.spatial_cspace, self.spatial_size)
		wi.extract_hist_features(self.hist_cspace, self.hist_bins, self.hist_range)
		wi.extract_hog_features(self.hog_cspace, self.hog_channel, self.orient, self.pix_per_cell, self.cell_per_block, False, False)
		wi.combine_features(self.spatial_f_en, self.color_hist_f_en, self.hog_f_en)
		q.put(self.std.transform(wi.image_features))
			