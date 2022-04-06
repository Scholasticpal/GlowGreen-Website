from traceback import print_tb
import cv2
import time
from sollutionChallenge.utils.ObjectDetectorOptions import *
import numpy as np
from threading import Thread

DETECTION_THRESHOLD = 0.1
options = ObjectDetectorOptions(num_threads=4,
                                score_threshold=DETECTION_THRESHOLD)
detector = ObjectDetector(model_path="sollutionChallenge/assets/apple_lr_1.0.0_beta.tflite", options=options)

class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id  # default is 0 for primary camera

        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.vcap stream
        self.stopped = True

        # reference to the thread for reading next available frame from input stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background while the program is executing

    # method for starting the thread for grabbing next available frame in input stream
    def start(self):
        self.stopped = False
        self.t.start()

        # method for reading next frame

    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()

    # method for returning latest read frame
    def read(self):
        return self.frame

    # method called to stop reading frames
    def stop(self):
        self.stopped = True

def gen_frames():
    webcam_stream = WebcamStream(stream_id=0)  # stream_id = 0 is for primary camera
    webcam_stream.start()
    while True:
        try:
            frame = webcam_stream.read()

            # adding a delay for simulating time taken for processing a frame
            # delay value in seconds. so, delay=1 is equivalent to 1 second

            image = cv2.resize(frame, (512, 512))
            image_np = np.asarray(image)
            detections = detector.detect(image_np)
            image_np = visualize(image_np, detections)
            frame = cv2.imencode('.jpg', image_np)[1]
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n'
            key = cv2.waitKey(1)


        except Exception as e:
            print(e)

    # printing time elapsed and fps

    # closing all windows

#     cap = cv2.VideoCapture(0)
#     while True:
#         success, img = cap.read()
#
#         try:
#             # cv2.imwrite('images.jpg', image)
#             image =cv2.resize(img, (512, 512))
#             # image = Image.open('images.jpg').convert('RGB')
#             # image.thumbnail((512, 512), Image.ANTIALIAS)
#             image_np = np.asarray(image)
#             detections = detector.detect(image_np)
#             image_np = visualize(image_np, detections)
#             # cv2.imshow("SALUCHAN",cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
#             frame = cv2.imencode('.jpg', image_np)[1]
#
#             # print(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#             yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n'
#
#             cv2.waitKey(1)
#
#         except Exception as e:
#             print(e)
#             break
#     cap.release()
#
#
# cv2.destroyAllWindows()


gen_frames()
