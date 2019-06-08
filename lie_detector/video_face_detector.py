import cv2
import numpy as np
import os

haarcascade_path = os.path.join('lie_detector', 'weights', 'haarcascade_frontalface_default.xml')


def generate_cropped_face_video(vpath, grayscale=False, fps=10):

    if vpath is None:
        return 0.0

    face_cascade = cv2.CascadeClassifier(haarcascade_path)

    cap = cv2.VideoCapture(vpath)
    inp_fps = cap.get(cv2.CAP_PROP_FPS)
    inp_frame_time = 1000.0/inp_fps
    frame_time = 1000.0/fps
    frame_time_counter = frame_time
    frame_available, img = cap.read()

    center = np.array([])
    dims = np.array([])
    samples = 0
    frames = 0

    while frame_available:

        # Logic to decrease fps
        if frame_time_counter >= frame_time:
            rect = _detect_face(img, face_cascade, center)
            frame_time_counter -= frame_time
            frames += 1

            if len(rect) > 0:
                
                center, dims = _update_rect(center, dims, rect, samples < 10)
                samples += 1

                if samples >= 10:

                    cropped_img = img[int(center[1]-dims[1]/2): int(center[1]+dims[1]/2), 
                                  int(center[0]-dims[0]/2): int(center[0]+dims[0]/2)]
                    # cropped_img = img[rect[1]:rect[1]+rect[3], rect[0]: rect[0]+rect[2]]

                    # cv2.imshow('cropped_vid', cropped_img)
                    # cv2.waitKey(50)

        frame_time_counter += inp_frame_time
        frame_available, img = cap.read()

    cap.release()
    return float(samples) / frames

def _update_rect(center, dims, rect, dims_sampling, smooth_coef=0.9):
    curr_center = rect[:2] + 0.5*rect[2:]
    
    if len(center)>0:
        center = smooth_coef*center+(1-smooth_coef)*curr_center
        if dims_sampling:
            dims = smooth_coef*dims+(1-smooth_coef)*rect[2:]
    else:
        center = curr_center
        dims = rect[2:]
    return center, dims

def _detect_face(img, face_cascade, prev_center, padding=20):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
    if len(faces) > 0:
        min_dist = 1e10
        best_face = faces[0]
        if len(prev_center) > 0:
            for f in faces:
                dist = (f[0]+f[2]/2-prev_center[0])**2 + (f[1]+f[1]/2-prev_center[1])**2
                if dist < min_dist:
                    min_dist = dist
                    best_face = f

        best_face[0] -= padding/2
        best_face[1] -= padding/2
        best_face[2] += padding
        best_face[3] += padding

        return np.array(best_face)
    else:
        return np.array([])

if __name__ == '__main__':
    print(generate_cropped_face_video('/home/ryan/Desktop/vid.mp4'))