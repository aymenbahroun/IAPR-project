##Special project
##created on May 1st
##
## Group 44 Deyanira Cisneros, Aymen Bahroun and Zeno Messi

#Load the Libraries
import imageio	#this library is made for handling the videos
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import fftpack
from scipy import ndimage

from skimage import filters
from skimage import util
import matplotlib.patches as mpatches
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage import segmentation
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import measure
from skimage import filters
from skimage import morphology
from skimage.transform import rescale, resize
from scipy.ndimage import rotate, shift
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
import progressbar
from time import sleep
from PIL import Image

import skimage
import math
import cmath

import gzip
import tarfile
import os

import argparse

parser = argparse.ArgumentParser(description='Read the video and output the equation')
parser.add_argument('--input', nargs=1, required=True, type=str)
parser.add_argument('--output', nargs=1, required=True, type=str)
args = parser.parse_args()
print(args.input[0], args.output[0])

def op_class(im, ext):
	err = []
	padded = np.pad(im, ((int(np.ceil((ext-im.shape[0])/2)),int(np.ceil((ext-im.shape[0])/2))), 
	(int(np.ceil((ext-im.shape[1])/2)), int(np.ceil((ext-im.shape[1])/2)))),  mode='constant', constant_values=0)
	for ang in range(0, 360):
		rot = rotate(padded, ang, reshape=False, order=3)
		err_ang = 0
		for i in range(padded.shape[0]):
			for j in range(padded.shape[1]):
				err_ang += (padded[i,j]-rot[i,j])**2
		err.append(err_ang)
	fouried = abs(fftpack.fft(np.asarray(err)))
	four_slice = fouried[1:10]
	return np.argmax(four_slice)+1
	
def digit_class(reg, im):
	mask = region.image<1
	treat = im
	treat[mask] = 0
	
	maximum = np.max(treat.shape)
	rescale_value = 1.
	if maximum > 20:
		rescale_value = 20./maximum
	
	rescaled_img = rescale(treat, rescale_value)
	padded = np.pad(rescaled_img, ((int(np.floor((28-rescaled_img.shape[0])/2)),int(np.ceil((28-rescaled_img.shape[0])/2))), 
				(int(np.floor((28-rescaled_img.shape[1])/2)), int(np.ceil((28-rescaled_img.shape[1])/2)))),  mode='constant', constant_values=0)
	M_rot = measure.moments(padded, order=3)
	padded = shift(padded, [np.round(14.-M_rot[1, 0]/M_rot[0,0]), np.round(14.-M_rot[0,1]/M_rot[0,0])])
	new_im = Image.fromarray(padded)
	new_im = img_to_array(new_im)
	new_im = new_im.reshape(1, 28, 28, 1)
	new_im = new_im.astype('float32')
	return new_im
	
	
def define_model():
	model = Sequential()   
	model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
	model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())
	model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
	model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(BatchNormalization())    
	model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(BatchNormalization())
	model.add(Dense(512,activation="relu"))
	model.add(Dense(9,activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
	return model
	
##open the video
video_path = args.input[0]	#the path for the video file
vid = imageio.get_reader(video_path, 'ffmpeg')	#opens the file

##infos about the video, like the frame rate, size and duration
metadata = vid.get_meta_data()
#print("metadata", metadata)

# create some variables with those info
nb_frames = int(metadata['duration']*metadata['fps'])
nb_rows = metadata['size'][1]
nb_cols = metadata['size'][0]
depth = 3

#create a numpy ndarray to put each frame. This array has size (number of frames, size in y, size in x, number of channels)
#this means here the size is (42, 480, 720, 3)
data_train = np.empty((int(metadata['duration']*metadata['fps']), metadata['size'][1], metadata['size'][0], 3), dtype='float64')
print("Shape of the image array", data_train.shape)

# This loop puts the images in the array
for num, image in enumerate(vid.iter_data()):
    data_train[num] = util.img_as_float64(image)

#Apply median filter to each frame
median = np.empty((nb_frames, data_train.shape[1], data_train.shape[2]))
for num, image in enumerate(data_train):
    median[num] = filters.median(image[:,:,0], behavior='ndimage')

#####CREATE THE MASKS FOR THE ROBOT
robot_masks = []
for i, im in enumerate(data_train):
    image_rev = data_train[i][:,:,2]
    image_rev = filters.median(image_rev[:,:], behavior='ndimage')
    image = util.invert(image_rev)

    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 1000:
            biggest_region = region

            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)

    minr, minc, maxr, maxc = biggest_region.bbox
    arrow_area_image = np.empty((data_train.shape[1], data_train.shape[2]), dtype='bool')
    for x in range(data_train.shape[1]):
        for y in range(data_train.shape[2]):
            if (x>=minr) & (x<=maxr) & (y>=minc) & (y<=maxc):
                arrow_area_image[x,y] = True
            else :
                arrow_area_image[x,y] = False
    robot_masks.append(arrow_area_image)

#####-------------

#####CREATE THE MASKS FOR THE OPERATIONS AND DIGITS
#Take the median of all the frames (median value for each pixel)
norobot = np.median(median, axis=0)

#####background normalization
mode = stats.mode(norobot, axis=None)[0][0]
for l in range(data_train.shape[1]):
	for j in range(data_train.shape[2]):
		norobot[l,j] = min(1., norobot[l,j]/mode)

##invert the image, we will need this image for the classification
norobot_inv = util.invert(norobot)

#####Remove dark bg and line in the middle and find regions

#Threshold

norobot_line = norobot[:] > 0.8
#Dilate the dark regions (i.e. erode the image)
norobot_line = morphology.binary_erosion(norobot_line, morphology.square(10))
# norobot_line=clear_border(util.invert(norobot_line), bgval=0)
# norobot_line=util.invert(norobot_line)
norobot_bin = norobot[:] > 0.8
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.imshow(norobot_line)
# plt.show();
# #Find regions
labels_line = measure.label(util.invert(norobot_line))
regions_line = measure.regionprops(labels_line)
# #Remove the biggest areas, i.e. line, borders
labels_toremove = []
for i, region in enumerate(regions_line):
	if region.area > 2000 :
		labels_toremove.append(i+1)
for lab in labels_toremove:
	for lin in range(len(labels_line)):
		for col in range(len(labels_line[lin])):
			if labels_line[lin,col] == lab:
				labels_line[lin,col] = 0
				norobot_bin[lin,col] = True
#update the labels' labels
regions_signs = measure.regionprops(labels_line)
region_nb = len(regions_signs)

#####Labels for classification.
labels = measure.label(util.invert(norobot_bin))
regions = measure.regionprops(labels)

divided = 0
equal = 0
for i, region in enumerate(regions_signs):
	
	labels_seg = measure.label(util.invert(norobot_bin[region.slice]))
	regions_seg = measure.regionprops(labels_seg)
	if len(regions_seg)==3:
		divided = i
	elif len(regions_seg)==2:
		equal = i
print(divided, equal)
for i,region in enumerate(regions_signs):
	for row, col in region.coords:
		if labels[row,col]!=0:
			labels[row,col] = labels_line[row,col]

regions = measure.regionprops(labels)

#####Load model
model = define_model()
model.load_weights("weights/best_norot.hdf5")
# compute max extension
max_ext = 0
for i in range(len(regions)):
	for j in range(2):
		if regions[i].image.shape[j]>max_ext:
			max_ext = regions[i].image.shape[j]

#####Dictionary for the operators
label_operators = {2:"-", 4:"+", 6:"*"}

operator_list = []

#####DO THE INTERSECTION
#####State of the robot, True is digit, False is operator
robot_state = True
just_seen = -1
operation_list = ''
operation_paste = []
positions = []
flag_end = False
for j, r_mask in enumerate(robot_masks):
	if flag_end:
		break
	for i,region in enumerate(regions):
		regions_bin = np.full(norobot.shape, False)
		regions_bin[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] = True
		area_region = np.count_nonzero(regions_bin)
		intersect = np.logical_and(regions_bin, r_mask)
		area_region = np.count_nonzero(regions_bin)
		area_intersect = np.count_nonzero(intersect)
		if area_intersect>0:
			if area_intersect//area_region==1 :
				if i==just_seen :
					continue
				if robot_state:
					predicted = np.argmax(model.predict(digit_class(region, norobot_inv[region.slice].copy())),axis=-1)[0]
					operation_paste.append((j, str(predicted)))
					operation_list += str(predicted)
					robot_state = False
					positions.append(j)
				else:
					robot_state = True
					if i==divided :
						operation_list += "/"
						operation_paste.append((j, "/"))
						
					elif i==equal :
						operation_paste.append((j,"="))
						flag_end = True
						positions.append(j)
						break
					else:
						predicted = label_operators[op_class(norobot_inv[region.slice].copy(), max_ext)]
						operation_paste.append((j,str(predicted)))
						operation_list += str(predicted)
					positions.append(j)
				just_seen = i
				print(operation_list)
print(operation_list)
final_result = eval(operation_list)
print(operation_list, "=", final_result)
operation_paste.append((operation_paste[-1][0], str(final_result)))
print(operation_paste)

# Outputting the video 

frames_path = "../data/new_frames/{i}.png"
writer = imageio.get_writer(args.output[0], fps=metadata['fps'])
centroids = []
equation = ''
position = 0

max_progress = np.asarray(data_train.shape)[0]
my_dpi = 107.
for i, im in enumerate(data_train):
    image_rev = im[:,:,2]/255
    image_rev = filters.median(image_rev[:,:], behavior='ndimage')
    image = util.invert(image_rev)
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    cleared = clear_border(bw)
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
    
    fig, ax =plt.subplots(figsize=(7.2, 4.8), dpi=my_dpi)
    ax.imshow(im)
    plt.axis('off')
    plt.margins(0)
    
    for region in regionprops(label_image):
        if region.area >= 1000:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
            y0, x0 = region.centroid
            centroids.append(region.centroid)
            
            if np.any(np.asarray(positions) == i):
                equation += ' '
                if np.asarray(operation_paste)[position,1] == '/':
                    equation += str(chr(247))
                    position += 1 
                else:
                    equation += np.asarray(operation_paste)[position,1]
                    if np.asarray(operation_paste)[position,1] == '=':
                        equation += ' '
                        equation += str(final_result)
                    position += 1
				
            if i == 0 :
                ax.plot(centroids[i][1], centroids[i][0], '.g', markersize=8)
                # plt.plot(centroids[i][1], centroids[i][0], '.g', markersize=8)
            else :
                for j in range(1,i):
                    dx = centroids[j-1][1] - centroids[j][1]
                    dy = centroids[j-1][0] - centroids[j][0]
                    ax.text(300, 460, equation, style='italic', bbox={'facecolor': 'gold', 'alpha': 0.5, 'pad': 10})
                    ax.arrow(centroids[j-1][1], centroids[j-1][0], -dx, -dy, color='green', head_width=6)
            fig.tight_layout()
            ax.set(xlim=[-0.5, 720-0.5], ylim=[480-0.5, -0.5], aspect=1)
            fig.savefig("../data/new_frames/{i}.png".format(i=i), bbox_inches='tight', transparent='True', pad_inches=0, dpi=my_dpi)

    writer.append_data(imageio.imread(frames_path.format(i=i)))
writer.close()


