from traceback import print_tb
import cv2
import time
from sollutionChallenge.utils.ObjectDetectorOptions import * 
from run import detector
from PIL import Image
import numpy as np
def gen_frames():
    cap = cv2.VideoCapture(0)
    fps=0
    while(cap.isOpened()):    
        start_time = time.time()
        success,img=cap.read() 

        if success==True:    
            image = cv2.flip(img,1)
            # img=cv2.resize(img,(0,0),fx=0.5,fy=0.5)
            cv2.putText(image, 'FPS: {:.2f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 20, 55), 1)
            # cv2.imshow("Emotion Recognition",image)
            frame = cv2.imencode('.jpg', image)[1]

            frame.thumbnail((512, 512), Image.ANTIALIAS)
            image_np = np.asarray(frame)
            detections = detector.detect(image_np)
            image_np = visualize(image_np, detections)
            # print(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image_np.tobytes() + b'\r\n')

            k = cv2.waitKey(1)
            fps= (1.0 / (time.time() - start_time))

            if k == ord('q'):
                break
        else:
            break    
    cap.release()  
cv2.destroyAllWindows() 

gen_frames()