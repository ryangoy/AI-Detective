import cv2


def generate_cropped_face_video(vpath, grayscale=False, fps=10):

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(vpath)
    inp_fps = cap.get(cv2.CAP_PROP_FPS)
    inp_frame_time = 1000.0/inp_fps
    frame_time = 1000.0/fps
    frame_time_counter = frame_time
    frame_available, img = cap.read()

    while frame_available:

        if frame_time_counter >= frame_time:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

            # for (x, y, w, h) in faces:
            #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cv2.imshow('img', img[y:y+h, x:x+w])
            
            cv2.waitKey(50)
            frame_time_counter -= frame_time

        frame_time_counter += inp_frame_time
        frame_available, img = cap.read()

    cap.release()


if __name__ == '__main__':
    generate_cropped_face_video('/home/ryan/Desktop/vid.mp4')