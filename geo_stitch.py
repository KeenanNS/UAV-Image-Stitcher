import os
import argparse
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from numpy import where
from numpy import linalg
import matplotlib.pyplot as plt
import psutil
import shutil

process = psutil.Process(os.getpid())
print(process.memory_info().rss)
N_FIRST_IMAGES = -1

NUM_ROWS = 3
class Organizer:
	def __init__(self, directory):
		self.directory = directory
		self.coords = []
		self.make_set()
		print('made_set')
	

		self.width = int(abs((max(self.coords[:,0]) - min(self.coords[:,0])) * 10e6))
		self.height = int(abs((max(self.coords[:,1]) - min(self.coords[:,1])) * 10e6))
		self.min_x_coord = min(self.coords[:,0])
		self.min_y_coord = min(self.coords[:,1])
		self.max_x_coord = max(self.coords[:,0])
		self.max_y_coord = max(self.coords[:,1])
		print('detecting rows')
		self.detect_rows()
		self.sort_by_time_and_gen_dict()
		print('encoding new file names')
		self.encode_file_names()
	
	def get_coords(self, image):
		#test image '../images/SERBI-190730-C3V1-0200.JPG'
		exif = Image.open(image)._getexif()
		time = exif[36867]
		numbers = re.findall('\d+', time)
		time = 3600 * int(numbers[-3]) + 60 * int(numbers[-2]) + int(numbers[-1])
		geo = exif[34853]
		lat = geo[2]
		latitude = lat[0][0] + lat[1][0]/60 + lat[2][0]/lat[2][1]/3600
		lon = geo[4]
		longitude = lon[0][0] + lon[1][0]/60 + lon[2][0]/lon[2][1]/3600
		return [longitude, latitude, time]	

	def make_set(self):
		images = []
		coords = []
		self.path_list = []
		for image in os.listdir(directory)[:N_FIRST_IMAGES]:
			if image.endswith('.JPG') and image.startswith('D'):
				self.path_list.append(image)
				coords = self.get_coords(directory+image)
				self.coords.append(coords)

		self.coords = np.asarray(self.coords)

	def sort_by_time_and_gen_dict(self):
		idxs = np.argsort(self.coords[:,2])
		self.image_dict = []
		for i in range(len(self.coords)):
			d = {}
			d['path'] = self.path_list[idxs[i]]
			d['row_ID'] = int(self.coords[idxs[i],3])
			d['time'] = int(self.coords[idxs[i], 2])		
			self.image_dict.append(d)
		del self.coords

	def more_vertical(self, m):
		if abs(m) > 1:
			return 1
		else:
			return 0

	def dist_between_point_and_line(self, m, b, point):
		return abs(m*point[0] - point[1] + b)/np.sqrt(m**2 + 1)

	def detect_rows(self):
		blank = np.zeros((self.width+1, self.height+1), dtype = np.uint8)
		# print(blank.shape)
		X = []
		Y = []
		dat = []
		for pair in self.coords:
			x = int((pair[0] - self.min_x_coord) * 10e6)
			X.append(x)
			y = int((pair[1]- self.min_y_coord) * 10e6)
			Y.append(y)
			blank[x][y] = 255
			dat.append([x,y])
		dat = np.asarray(dat)

		m, b = np.polyfit(self.coords[:,0], self.coords[:,1], 1)
		# converting the line to be one of the lines on the extremity of the dataset
		NUM_ROWS = 3
		row_ID = np.zeros((len(self.coords),1))
		if self.more_vertical(m):
			root_points = self.find_n_leftmost()
			for line_idx, root in enumerate(root_points):
				point = self.coords[root][:2]
				b = point[1] - m * point[0]
				for i, coord in enumerate(self.coords):
					if self.dist_between_point_and_line(m, b, coord[:2]) < 0.00004:
						row_ID[i] = line_idx
	
		else:
			root_points = self.find_n_lowest()
			for line_idx, root in enumerate(root_points):
				point = self.coords[root][:2]
				b = point[1] - m * point[0]
				for i, coord in enumerate(self.coords):
					if self.dist_between_point_and_line(m, b, coord[:2]) < 0.00003:
						row_ID[i] = line_idx
						#self.coords[i] = np.concatenate(self.coords[i],line_idx)
		
		self.coords = np.append(self.coords, row_ID, axis = 1)
		colours = ['red', 'blue', 'green','black']
		colors = [colours[int(i)] for i in self.coords[:,3]]
		#plt.scatter(self.coords[:,0], self.coords[:,1], color = colors)
		x = [self.min_x_coord, self.max_x_coord]

	def find_n_lowest(self):
		return np.argsort(self.coords[:,1])[:NUM_ROWS]

	def find_n_leftmost(self):
		return np.argsort(self.coords[:,0])[:NUM_ROWS]
		
	def direction(self, point1, point2):
		x = point1[0]- point2[0]
		y = point1[1] - point2[1]
		return np.arctan(y/x)

	def get_time_taken(img):
		return Image.open(img).getexif()[36867]
###encode all the info into the filename
	def encode_file_names(self):
		if not os.path.exists('test_folder'):
			os.mkdir('test_folder')
		dest_dir = os.getcwd()+'/test_folder'
		for image in self.image_dict:
			shutil.copy(os.path.join(directory,image['path']),dest_dir)
			dst_file = os.path.join(dest_dir,image['path'])
			new_dst_file_name = os.path.join(dest_dir, str(image['time'])+'-'+str(image['row_ID'])+'.JPG')
			os.rename(dst_file, new_dst_file_name)


class Matcher:
	def __init__(self):
		print('matcher init')
		self.SIFT = cv2.SIFT_create()
		self.matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})
		self.step_dict = {
		'row' : [None, 3600, None, None],
		'batch' : [None, 3600, -3600, None]
		}
	def to_BW(self, image):
		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	def filter_matches(self, matches, ratio = 0.75):
		filtered_matches = []
		for m in matches:
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				filtered_matches.append(m[0])

		return filtered_matches

	def imageDistance(self, matches):
		sumDist = 0
		for match in matches:
			sumDist += match.distance
		return sumDist

	def findDimensions(self, image, homography):
		base_p1 = np.ones(3, np.float32)
		base_p2 = np.ones(3, np.float32)
		base_p3 = np.ones(3, np.float32)
		base_p4 = np.ones(3, np.float32)
	
		(y, x) = image.shape[:2]
	
		base_p1[:2] = [0,0]
		base_p2[:2] = [x,0]
		base_p3[:2] = [0,y]
		base_p4[:2] = [x,y]
	
		max_x = None
		max_y = None
		min_x = None
		min_y = None
	
		for pt in [base_p1, base_p2, base_p3, base_p4]:
	
			hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T
	
			hp_arr = np.array(hp, np.float32)
	
			normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)
	
			if ( max_x == None or normal_pt[0,0] > max_x ):
				max_x = normal_pt[0,0]
	
			if ( max_y == None or normal_pt[1,0] > max_y ):
				max_y = normal_pt[1,0]
	
			if ( min_x == None or normal_pt[0,0] < min_x ):
				min_x = normal_pt[0,0]
	
			if ( min_y == None or normal_pt[1,0] < min_y ):
				min_y = normal_pt[1,0]
	
		min_x = min(0, min_x)
		min_y = min(0, min_y)
	
		return (min_x, min_y, max_x, max_y)
	def reduce_big_image(self, last_image_idx, new_image_idx):
		pass

	def match2(self, img1, img2, step):
		#need rgb and bw versions of both images
		Slice = self.step_dict[step]
		base_image = self.to_BW(img1)
		next_image = self.to_BW(img2)
		print('SIFT detect and compute')
		self.mem.append(self.process.memory_info().rss)
		self.Vmem.append(self.process.memory_info().vms)
		base_features, base_descriptions = self.SIFT.detectAndCompute(base_image[Slice[0]:Slice[1]], None)
		new_features, new_descriptions = self.SIFT.detectAndCompute(next_image[Slice[2]:Slice[3]], None)
		self.mem.append(self.process.memory_info().rss)
		self.Vmem.append(self.process.memory_info().vms)
		print('knn Matching')
		matches = self.matcher.knnMatch(new_descriptions, trainDescriptors = base_descriptions, k = 2)
		print('number of matches: ', len(matches))
		
		matches_subset = self.filter_matches(matches)
		distance = self.imageDistance(matches_subset)
		
		kp1 = []
		kp2 = []

		for match in matches_subset:
			kp1.append(base_features[match.trainIdx])
			kp2.append(new_features[match.queryIdx])
		
		p1 = np.array([k.pt for k in kp1])
		p2 = np.array([k.pt for k in kp2])

		H, stat = cv2.findHomography(p1,p2, cv2.RANSAC, 5.0)
		inlierRatio = float(np.sum(stat)) / float(len(stat))
		print('inlier ratio: ', inlierRatio)
		H = H / H[2,2]
		H_inv = linalg.inv(H)
		(min_x, min_y, max_x, max_y) = self.findDimensions(img2, H_inv)

		# Adjust max_x and max_y by base img size
		max_x = max(max_x, base_image.shape[1])
		max_y = max(max_y, base_image.shape[0])

		move_h = np.matrix(np.identity(3), np.float32)

		if ( min_x < 0 ):
			move_h[0,2] += -min_x
			max_x += -min_x

		if ( min_y < 0 ):
			move_h[1,2] += -min_y
			max_y += -min_y


		mod_inv_h = move_h * H_inv

		img_w = int(math.ceil(max_x))
		img_h = int(math.ceil(max_y))

		base_h, base_w, base_d = img1.shape
		next_h, next_w, next_d = img2.shape

		img1 = img1[5:(base_h-5),5:(base_w-5)]
		img2 = img2[5:(next_h-5),5:(next_w-5)]
		base_img_warp = cv2.warpPerspective(img1, move_h, (img_w, img_h))
		next_img_warp = cv2.warpPerspective(img2, mod_inv_h, (img_w, img_h))

		enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

		(ret,data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY), 
			0, 255, cv2.THRESH_BINARY)

		# add base image
		enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, 
			mask=np.bitwise_not(data_map), 
			dtype=cv2.CV_8U)

		# add next image
		final_img = cv2.add(enlarged_base_img, next_img_warp, 
			dtype=cv2.CV_8U)

		#Crop black edge
		final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
		_, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		max_area = 0
		best_rect = (0,0,0,0)

		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt)

			deltaHeight = h-y
			deltaWidth = w-x

			area = deltaHeight * deltaWidth

			if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
				max_area = area
				best_rect = (x,y,w,h)

		if ( max_area > 0 ):
			final_img_crop = final_img[best_rect[1]:best_rect[1]+best_rect[3],
					best_rect[0]:best_rect[0]+best_rect[2]]

			final_img = final_img_crop
		return final_img, img_w, x, w

class traffic_director(Organizer, Matcher):
	def __init__(self, directory):
		Matcher.__init__(self)
		self.mem = []
		self.Vmem = []
		self.process = psutil.Process(os.getpid())
		self.feed_matcher()
	
	def get_spans(self, stitched_top, bottom_part):
		lower = (1,1,1)
		upper = (255,255,255)

		bottom_row_of_top = stitched_top[-2:]
		top_row_of_bottom = bottom_part[:2] 

		thresh = cv2.inRange(bottom_row_of_top, lower, upper)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		top_image_span = [cv2.boundingRect(contours[-1])[0], cv2.boundingRect(contours[0])[0] + cv2.boundingRect(contours[0])[2]]

		thresh = cv2.inRange(top_row_of_bottom ,lower, upper)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		bottom_image_span = [cv2.boundingRect(contours[-1])[0], cv2.boundingRect(contours[0])[0] + cv2.boundingRect(contours[0])[2]]
		return top_image_span, bottom_image_span

	def feed_matcher(self):
		self.row_1 = []
		for image in os.listdir('test_folder'):
			image = 'test_folder/'+image
			row_idx = image[-5]
			if row_idx == '0':
				self.row_1.append(image)
		self.row_1.sort()
		return
		batch_size = 10
		for batch in range(len(row_1)//batch_size):	
			img1 = cv2.imread(row_1[batch*batch_size])
			batch_end = (batch * batch_size) + batch_size
			if batch_end < len(row_1):
				end = batch_end
			else :
				end = len(row_1)-1
			for i, image in enumerate(row_1[batch * batch_size + 1:end]):	
				self.mem.append(self.process.memory_info().rss)
				self.Vmem.append(self.process.memory_info().vms)
				print('IMAGE NUMBER: ', batch * batch_size +i+1)
				img2 = cv2.imread(image)
				img1 = self.match2(img1, img2, 'row')
			
			cv2.imwrite('batches/image'+str(batch)+'.jpg', img1)	
		batch_list = []
		for image in os.listdir('batches'):
			image = 'batches'+image
			batch_list.append(image)
		batch_list.sort()
		img1 = cv2.imread(batch_list[0])
		for i, image in enumerate(batch_list[1:]):
			img2 = cv2.imread(batch_list[i])
			img1 = self.match2(img1, img2, 'batch')
			top_part = img1[:3600]
			bottom_part = img2[-3600:]
			stiched_top = self.match2(top_part,im2, 'batch')
			# np.concatenate()
			cv2.imwrite('batchfinal.jpg', img1)

		##consider just cutting the image in half with a clean line and keeping track. 

######### Running section #############
directory = '../images/new_set/'
Organizer(directory)
#org = Organizer(directory)
test = traffic_director(directory)
img2 = cv2.imread(test.row_1[10])
# plt.imshow(img2)
# plt.show()
img1 = cv2.imread('batches/image0.jpg')
# plt.imshow(img1)
# plt.show()
top_part = img1[:3600]
bottom_part = img1[3600:]

# print('top part')
# plt.imshow(top_part)
# plt.show()

# print('bottom part')
# plt.imshow(bottom_part)
# plt.show()
stitched_top,img_w, x, w = test.match2(top_part,img2, 'batch')
# enlarged_base_img = np.zeros((bottom_part.shape[1], img_w, 3), np.uint8)
lower = (1,1,1)
upper = (255,255,255)

bottom_row_of_top = stitched_top[-2:]
top_row_of_bottom = bottom_part[:2] 

thresh = cv2.inRange(bottom_row_of_top, lower, upper)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
top_image_span = [cv2.boundingRect(contours[-1])[0], cv2.boundingRect(contours[0])[0] + cv2.boundingRect(contours[0])[2]]

thresh = cv2.inRange(top_row_of_bottom ,lower, upper)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
bottom_image_span = [cv2.boundingRect(contours[-1])[0], cv2.boundingRect(contours[0])[0] + cv2.boundingRect(contours[0])[2]]

print(top_image_span, bottom_image_span)
top_dims = stitched_top.shape
bottom_dims = bottom_part.shape
left_add = bottom_image_span[0]-top_image_span[0]
left_pad = np.zeros((top_dims[0], left_add,3), np.uint8)
right_add = left_add + top_dims[1] - bottom_dims[1]
right_pad = np.zeros((bottom_dims[0], right_add,3), np.uint8)
stitched_top = np.concatenate((left_pad,stitched_top), axis = 1)
plt.imshow(stitched_top)
plt.show()
bottom_part = np.concatenate((bottom_part, right_pad), axis = 1)
plt.imshow(bottom_part)
plt.show()
final_img = np.concatenate((stitched_top, bottom_part))
print(stitched_top.shape)
print(bottom_part.shape)
plt.imshow(final_img)
plt.show()
print(top_dims, bottom_dims)
print(left_pad)
print(right_pad)
exit()






