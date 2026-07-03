import os

import cv2
import mediapipe as mp
import numpy as np
from matplotlib import path
from scipy import interpolate


def get_mediapipe_landmarks(image_bgr):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=3,
        refine_landmarks=False,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    try:
        results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    finally:
        face_mesh.close()

    if not results.multi_face_landmarks:
        return None

    scores = []
    for face_landmarks in results.multi_face_landmarks:
        scores.append(abs(face_landmarks.landmark[386].x - face_landmarks.landmark[159].x))
    face_landmarks = results.multi_face_landmarks[int(np.argmax(scores))]

    pts = np.zeros((468, 3), dtype=np.float32)
    for i in range(468):
        pts[i, 0] = face_landmarks.landmark[i].x
        pts[i, 1] = face_landmarks.landmark[i].y
        pts[i, 2] = face_landmarks.landmark[i].z
    return pts


def load_triangles(path_or_dir):
    if os.path.isdir(path_or_dir):
        path_or_dir = os.path.join(path_or_dir, "triangles.txt")
    values = np.loadtxt(path_or_dir, usecols=1, dtype=float)
    return np.reshape(values, (-1, 3)).astype(np.int32)


def landmarks_to_pixels(pts3d, image):
    h, w = image.shape[:2]
    pts2d = np.transpose(np.vstack((pts3d[:, 0] * w, pts3d[:, 1] * h))).astype(np.float32)
    pts2d[:, 0] = np.clip(pts2d[:, 0], 0, w - 1)
    pts2d[:, 1] = np.clip(pts2d[:, 1], 0, h - 1)
    return pts2d


def draw_landmarks(image, pts2d):
    out = image.copy()
    for x, y in pts2d:
        cv2.circle(out, (int(x), int(y)), 1, (0, 0, 255), -1)
    return out


def warp_by_triangles(source, target, source_pts2d, target_pts2d, triangles):
    output = np.zeros_like(target)
    target_h, target_w = target.shape[:2]
    num_pixels = target_h * target_w
    inds = np.zeros((num_pixels,), dtype=np.float64)
    source_samples = np.zeros((2, num_pixels), dtype=np.float64)
    cnt = 0

    for tri in triangles:
        src_tri = np.array([[source_pts2d[int(i), 0], source_pts2d[int(i), 1]] for i in tri], dtype=np.float32)
        dst_tri = np.array([[target_pts2d[int(i), 0], target_pts2d[int(i), 1]] for i in tri], dtype=np.float32)
        if abs(cv2.contourArea(dst_tri)) < 1e-4:
            continue

        warp_mat = cv2.getAffineTransform(dst_tri, src_tri)
        xmin = max(int(np.floor(dst_tri[:, 0].min())), 0)
        xmax = min(int(np.ceil(dst_tri[:, 0].max())), target_w - 1)
        ymin = max(int(np.floor(dst_tri[:, 1].min())), 0)
        ymax = min(int(np.ceil(dst_tri[:, 1].max())), target_h - 1)
        if xmax < xmin or ymax < ymin:
            continue

        xx, yy = np.mgrid[xmin : xmax + 1, ymin : ymax + 1]
        xx = xx.flatten()
        yy = yy.flatten()
        candidates = np.transpose(np.vstack((xx, yy)))
        mask = path.Path(dst_tri).contains_points(candidates)
        if not np.any(mask):
            continue

        selected_x = xx[mask]
        selected_y = yy[mask]
        ind = selected_x * target_h + selected_y
        sample = warp_mat @ np.vstack((selected_x, selected_y, np.ones((1, ind.shape[0]))))

        n = ind.shape[0]
        if cnt + n > num_pixels:
            n = num_pixels - cnt
            ind = ind[:n]
            sample = sample[:, :n]
        inds[cnt : cnt + n] = ind
        source_samples[:, cnt : cnt + n] = sample
        cnt += n
        if cnt >= num_pixels:
            break

    inds = inds[:cnt]
    source_samples = source_samples[:, :cnt]
    if cnt == 0:
        raise RuntimeError("No valid warped pixels")

    source_h, source_w = source.shape[:2]
    source_samples[0] = np.clip(source_samples[0], 0, source_w - 1)
    source_samples[1] = np.clip(source_samples[1], 0, source_h - 1)

    for ch in range(3):
        interp = interpolate.RegularGridInterpolator(
            (np.arange(0, source_w), np.arange(0, source_h)),
            np.transpose(source[:, :, ch]),
            bounds_error=False,
            fill_value=0,
        )
        output[(inds % target_h).astype(int), (inds // target_h).astype(int), ch] = interp(np.transpose(source_samples))

    return output


def mediapipe_warp_face(source_bgr, target_bgr, triangles_path):
    triangles = load_triangles(triangles_path)
    source_pts3d = get_mediapipe_landmarks(source_bgr)
    target_pts3d = get_mediapipe_landmarks(target_bgr)
    if source_pts3d is None:
        raise RuntimeError("MediaPipe could not detect a face in source")
    if target_pts3d is None:
        raise RuntimeError("MediaPipe could not detect a face in target")

    source_pts2d = landmarks_to_pixels(source_pts3d, source_bgr)
    target_pts2d = landmarks_to_pixels(target_pts3d, target_bgr)
    warped = warp_by_triangles(
        source_bgr.astype(np.float32),
        target_bgr.astype(np.float32),
        source_pts2d,
        target_pts2d,
        triangles,
    )
    return np.clip(warped, 0, 255).astype(np.uint8), draw_landmarks(target_bgr, target_pts2d)

