from traceback import print_tb
import cv2
import time
from sollutionChallenge.utils.ObjectDetectorOptions import *
from run import detector
from PIL import Image
import numpy as np


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()

        try:
            # cv2.imwrite('images.jpg', image)
            image =cv2.resize(img, (512, 512))
            # image = Image.open('images.jpg').convert('RGB')
            # image.thumbnail((512, 512), Image.ANTIALIAS)
            image_np = np.asarray(image)
            detections = detector.detect(image_np)
            image_np = visualize(image_np, detections)
            # cv2.imshow("SALUCHAN",cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            frame = cv2.imencode('.jpg', image_np)[1]

            # print(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n'

            cv2.waitKey(1)

        except Exception as e:
            print(e)
            break
    cap.release()


cv2.destroyAllWindows()

gen_frames()
