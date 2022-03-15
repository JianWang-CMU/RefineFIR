import cv2 as cv
import mediapipe as mp
import scipy.io as sio
import numpy as np
import glob
import natsort
import os
from pathlib import Path
import timeit

# user specifies filenames here
file_extension = 'jpeg'
results_folder = 'result'
# end

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

Path(results_folder).mkdir(parents=True, exist_ok=True)
max_numoffaces = 1
bool_notracking = True
pts3D = np.zeros((468, 3))
#
# For static images:
filenames = glob.glob('*.' + file_extension)
IMAGE_FILES = natsort.natsorted(filenames)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=bool_notracking,
    max_num_faces=max_numoffaces,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    print(file)
    image = cv.imread(file)
    # Convert the BGR image to RGB before processing.
    start = timeit.default_timer()
    results = face_mesh.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      for i in range(468):
        pts3D[i, 0] = face_landmarks.landmark[i].x
        pts3D[i, 1] = face_landmarks.landmark[i].y
        pts3D[i, 2] = face_landmarks.landmark[i].z

      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)
    sio.savemat(os.path.splitext(file)[0] + '_landmarks_mediapipe.mat', {'pts3D': pts3D})
    cv.imwrite(results_folder + '/' + os.path.splitext(file)[0] + '.jpg', annotated_image)