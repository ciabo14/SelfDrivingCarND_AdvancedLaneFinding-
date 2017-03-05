from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt

import numpy as np
class RoadImage:

	def __init__(self, image):
		self.image = image
		self.heatmap = np.zeros_like(image)
		self.hog_features = []
		self.hog_features_1 = []
		self.hog_features_2 = []
		self.hog_features_3 = []
		
		self.windows = []
		self.windows_images = np.array([])
		self.windows_detected = np.array([])
	
	def change_color_space(self, cspace):
	
		if cspace != 'RGB':
			if cspace == 'HSV':
				feature_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
			elif cspace == 'LUV':
				feature_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2LUV)
			elif cspace == 'HLS':
				feature_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)
			elif cspace == 'YUV':
				feature_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV)
			elif cspace == 'YCrCb':
				feature_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2YCrCb)
		else:
			feature_image = np.copy(self.image)
			
		return feature_image
	
	def extract_hog_features(self, cspace = 'RGB', hog_channel='ALL', orient=9, pix_per_cell=8, 
							cell_per_block=2, vis=False, feature_vec=False):
							
		feature_image = self.change_color_space(cspace)

		if hog_channel == 'ALL':
			hog_features = []
			self.hog_features_1 = self.get_hog_features(feature_image[:,:,0], orient, pix_per_cell, cell_per_block, vis, feature_vec)
			self.hog_features_2 = self.get_hog_features(feature_image[:,:,1], orient, pix_per_cell, cell_per_block, vis, feature_vec)
			self.hog_features_3 = self.get_hog_features(feature_image[:,:,2], orient, pix_per_cell, cell_per_block, vis, feature_vec)
		
		else:
			self.hog_features = hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis, feature_vec)
		
	def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
			
		if vis:
			features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
									  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
									  visualise=vis, feature_vector=feature_vec)
			return features, hog_image
		else:
			features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
									  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
									  visualise=vis, feature_vector=feature_vec)
			return features
			
	def get_window(self, top_left_corner,bottom_right_corner):
		return self.image[top_left_corner[0]:bottom_right_corner[0],top_left_corner[1],bottom_right_corner[1]]