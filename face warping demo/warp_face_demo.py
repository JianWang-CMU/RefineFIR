import cv2 as cv
import os
import numpy as np
import scipy.io as sio
from matplotlib import path
from scipy import interpolate
import time
import os

# specify filenames here
file_imSource = 'rihanna1.jpeg'
file_imTarget = 'rihanna2.jpeg'
# file_imTarget = ''

results_folder = 'result'
# end

# load data
# read txt for xyz <-> uv
val = np.loadtxt('canonical face mesh/geometry_pipeline_metadata_landmarks.pbtxt1.txt', usecols=1, skiprows=0, dtype=float)
val = np.transpose(np.reshape(val, (-1, 5)))
pt3d = val[0:3, :]
uv = val[3:, :]

val = np.loadtxt('canonical face mesh/geometry_pipeline_metadata_landmarks.pbtxt2.txt', usecols=1, skiprows=0, dtype=float)
T = np.reshape(val, (-1, 3))

# imSource
f = sio.loadmat(os.path.splitext(os.path.basename(file_imSource))[0] + '_landmarks_mediapipe.mat')
im_pt3d = f['pts3D']
im = cv.imread(file_imSource)
imh, imw = im.shape[0], im.shape[1]
im_pt2d = np.transpose(np.vstack((im_pt3d[:,0]*imw, im_pt3d[:,1]*imh)))
imSource = im
imSource_pt2d = im_pt2d

# imTarget
if file_imTarget != '':
	f = sio.loadmat(os.path.splitext(os.path.basename(file_imTarget))[0] + '_landmarks_mediapipe.mat')
	im_pt3d = f['pts3D']
	im = cv.imread(file_imTarget)
	imh, imw = im.shape[0], im.shape[1]
	im_pt2d = np.transpose(np.vstack((im_pt3d[:,0]*imw, im_pt3d[:,1]*imh)))
	imTarget = im
	imTarget_pt2d = im_pt2d
else:
	mapH = 800
	mapW = 800
	imTarget = np.zeros((mapH, mapW, 3))
	imTarget_pt2d = np.transpose(np.vstack((uv[0,:]*mapW, uv[1,:]*mapH)))

def warpFace(imSource, imTarget, imSource_pt2d, imTarget_pt2d, T):
	n = T.shape[0]

	imTarget_H, imTarget_W = imTarget.shape[0], imTarget.shape[1]
	xx, yy = np.mgrid[0:imTarget_W, 0:imTarget_H]
	xx = xx.flatten()
	yy = yy.flatten()
	xxyy = np.transpose(np.vstack((xx, yy)))

	numPxls = imTarget_H * imTarget_W
	inds = np.zeros((numPxls,))
	target_triangles = np.zeros((2, numPxls))
	# IND = np.arange(numPxls)
	cnt = 0
	start_time = time.time()

	for i in range(n):
		srcTri = np.array([[imSource_pt2d[int(T[i, 0]), 0], imSource_pt2d[int(T[i, 0]), 1]], \
						   [imSource_pt2d[int(T[i, 1]), 0], imSource_pt2d[int(T[i, 1]), 1]], \
						   [imSource_pt2d[int(T[i, 2]), 0], imSource_pt2d[int(T[i, 2]), 1]]]).astype(np.float32)
		dstTri = np.array([[imTarget_pt2d[int(T[i, 0]), 0], imTarget_pt2d[int(T[i, 0]), 1]], \
						   [imTarget_pt2d[int(T[i, 1]), 0], imTarget_pt2d[int(T[i, 1]), 1]], \
						   [imTarget_pt2d[int(T[i, 2]), 0], imTarget_pt2d[int(T[i, 2]), 1]]]).astype(np.float32)

		warp_mat = cv.getAffineTransform(dstTri, srcTri)

		# start_time = time.time()
		# p = path.Path(dstTri)
		# inMask = p.contains_points(xxyy)
		# # print("--- %s seconds ---" % (time.time() - start_time))
		# ind = IND[inMask]
		# target_triangle = warp_mat @ np.vstack((xx[inMask], yy[inMask], np.ones((1, ind.shape[0]))))

		# start_time = time.time()
		# another method: by three lines manually, still slow
		# a0 = 0
		# a1 = 1
		# a2 = 2
		# a = (dstTri[a0, 1] - dstTri[a2, 1]) * (dstTri[a1, 0] - dstTri[a2, 0]) - (dstTri[a1, 1] - dstTri[a2, 1]) * (dstTri[a0, 0] - dstTri[a2, 0])
		# aa = (xxyy[:,1] - dstTri[a2, 1]) * (dstTri[a1, 0] - dstTri[a2, 0]) - (dstTri[a1, 1] - dstTri[a2, 1]) * (xxyy[:,0] - dstTri[a2, 0])
		# a0 = 1
		# a1 = 0
		# a2 = 2
		# b = (dstTri[a0, 1] - dstTri[a2, 1]) * (dstTri[a1, 0] - dstTri[a2, 0]) - (dstTri[a1, 1] - dstTri[a2, 1]) * (
		# 			dstTri[a0, 0] - dstTri[a2, 0])
		# bb = (xxyy[:,1]- dstTri[a2, 1]) * (dstTri[a1, 0] - dstTri[a2, 0]) - (dstTri[a1, 1] - dstTri[a2, 1]) * (
		# 			xxyy[:,0] - dstTri[a2, 0])
		# a0 = 2
		# a1 = 0
		# a2 = 1
		# c = (dstTri[a0, 1] - dstTri[a2, 1]) * (dstTri[a1, 0] - dstTri[a2, 0]) - (dstTri[a1, 1] - dstTri[a2, 1]) * (
		# 			dstTri[a0, 0] - dstTri[a2, 0])
		# cc = (xxyy[:,1] - dstTri[a2, 1]) * (dstTri[a1, 0] - dstTri[a2, 0]) - (dstTri[a1, 1] - dstTri[a2, 1]) * (
		# 			xxyy[:,0] - dstTri[a2, 0])
		# # print("--- %s seconds ---" % (time.time() - start_time))
		# # inMask = (aa * a >= 0) * (bb * b >= 0) * (cc * c >= 0)
		# inMask = ((aa >= 0) == (a >= 0)) & ((bb >= 0) == (b >= 0)) & (((cc >= 0) & (c >= 0)))
		# # print("--- %s seconds ---a" % (time.time() - start_time))

		# ind = IND[inMask]
		# target_triangle = warp_mat @ np.vstack((xx[inMask], yy[inMask], np.ones((1, ind.shape[0]))))

		# another method: only the ROI
		# start_time = time.time()
		xmin = int(dstTri[:, 0].min())
		xmax = int(dstTri[:, 0].max())
		ymin = int(dstTri[:, 1].min())
		ymax = int(dstTri[:, 1].max())
		xx_, yy_ = np.mgrid[xmin:xmax+1, ymin:ymax+1]
		xx_ = xx_.flatten()
		yy_ = yy_.flatten()
		xxyy_ = np.transpose(np.vstack((xx_, yy_)))
		p_ = path.Path(dstTri)
		inMask_ = p_.contains_points(xxyy_)
		# print("--- %s seconds ---a" % (time.time() - start_time))
		xxyy_selected_x = xx_[inMask_]
		xxyy_selected_y = yy_[inMask_]
		ind = xxyy_selected_x * imTarget_H + xxyy_selected_y
		target_triangle = warp_mat @ np.vstack((xxyy_selected_x, xxyy_selected_y, np.ones((1, ind.shape[0]))))

		l = ind.shape[0]
		inds[cnt:cnt + l] = ind
		target_triangles[:, cnt:cnt + l] = target_triangle
		cnt = cnt + l

	print("--- %s seconds ---" % (time.time() - start_time))

	inds = np.delete(inds, np.arange(cnt, numPxls))
	target_triangles = np.delete(target_triangles, np.arange(cnt, numPxls), 1)
	imSource_H, imSource_W = imSource.shape[0], imSource.shape[1]
	for ch in range(3):
		f = interpolate.RegularGridInterpolator((np.arange(0, imSource_W), np.arange(0, imSource_H)), np.transpose(imSource[:, :, ch]))
		imTarget[(inds % imTarget_H).astype(int), (inds // imTarget_H).astype(int), ch] = f(np.transpose(target_triangles))

	return imTarget

imTarget = warpFace(imSource, imTarget, imSource_pt2d, imTarget_pt2d, T)
cv.imshow('sample image', imTarget/255)
cv.waitKey(0)
cv.imwrite(results_folder + '/' + os.path.splitext(file_imTarget)[0] + '_warped.jpeg', imTarget)