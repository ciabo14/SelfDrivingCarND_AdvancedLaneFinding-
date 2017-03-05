from skimage.feature import hog
import cv2

import numpy as np
class WindowImage:

	def __init__(self, image, hog_features = np.array([])):
		self.image = image
		self.spatial_features = np.array([])
		self.hist_features = np.array([])
		self.hog_features = hog_features
		self.image_features = np.array([])
		
	# Execute feature extraction depending on the parameters spatial_f_en, color_hist_f_en, hog_f_en
	def extract_features(self,  cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',spatial_size = 32,
						hist_bins = 20, hist_range = 10,
						spatial_f_en = True, color_hist_f_en= True, hog_f_en= True):
		#Convert the original RGB image in the desired color space
		feature_image = self.change_color_space(cspace)
		
		# Apply bin_spatial() to get spatial color features
		if spatial_f_en:
			self.spatial_features = self.bin_spatial(feature_image, size=(spatial_size,spatial_size))
		# Apply color_hist() also with a color space option now
		if color_hist_f_en:
			self.hist_features = self.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
			
		if hog_f_en:
			# Apply hog feature extraction
			self.hog_features = self.compute_HOG(feature_image, hog_channel, orient, pix_per_cell, cell_per_block, 
												vis=False, feature_vec=True)
		# Append the new feature vector to the features list
		self.image_features = (np.concatenate((self.spatial_features, self.hist_features, self.hog_features)))

	def combine_features(self, spatial_f_en = True, color_hist_f_en= True, hog_f_en= True):
		self.image_features = []
		if(spatial_f_en):
			self.image_features = (np.concatenate((self.image_features,self.spatial_features)))
		if(color_hist_f_en):
			self.image_features = (np.concatenate((self.image_features,self.hist_features)))
		if(hog_f_en):
			self.image_features = (np.concatenate((self.image_features,self.hog_features)))
			
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
			#elif cspace == 'GRAY':
			#	feature_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2YCrCb)
		else:
			feature_image = np.copy(self.image)
			
		return feature_image
	
	def extract_spatial_features(self, cspace = 'RGB' , spatial_size = 32):
		feature_image = self.change_color_space(cspace)
		self.spatial_features = self.bin_spatial(feature_image, size=(spatial_size,spatial_size))
	
	def extract_hist_features(self, cspace = 'RGB' , hist_bins = 20, hist_range = (0,256)):
		feature_image = self.change_color_space(cspace)
		self.hist_features = self.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
		
	def extract_hog_features(self, cspace = 'RGB', hog_channel='ALL', orient=9, pix_per_cell=8, 
							cell_per_block=2, vis=False, feature_vec=False):
		feature_image = self.change_color_space(cspace)
		self.hog_features = self.compute_HOG(feature_image, hog_channel, orient, pix_per_cell, cell_per_block,vis, feature_vec)
		
	# Execute hog feature extraction and save the result in the self.hog_features list
	def bin_spatial(self, feature_image, size=(32, 32)):
		return cv2.resize(feature_image, size).ravel() 

	# Execute hog feature extraction and save the result in the self.hog_features list
	def color_hist(self, feature_image, nbins=32, bins_range=(0, 256)):
		# Compute the histogram of the RGB channels separately
		rhist = np.histogram(feature_image[:,:,0], bins=nbins, range=bins_range)
		ghist = np.histogram(feature_image[:,:,1], bins=nbins, range=bins_range)
		bhist = np.histogram(feature_image[:,:,2], bins=nbins, range=bins_range)
		# Generating bin centers
		bin_edges = rhist[1]
		bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
		# Concatenate the histograms into a single feature vector
		hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
		# Return the individual histograms, bin_centers and feature vector
		return hist_features
		
	
	# Execute hog feature extraction and save the result in the self.hog_features list
	def compute_HOG(self, feature_image, hog_channel,orient, pix_per_cell, cell_per_block, vis, feature_vec):
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.append(self.get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, vis, feature_vec))
			hog_features = np.ravel(hog_features)        
		else:
			hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis, feature_vec)
			
		return hog_features
		
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