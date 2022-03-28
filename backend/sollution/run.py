from sollutionChallenge import app
from sollutionChallenge.utils.ObjectDetectorOptions import *
DETECTION_THRESHOLD = 0.1
options = ObjectDetectorOptions(num_threads=4,
                                score_threshold=DETECTION_THRESHOLD)
detector = ObjectDetector(model_path="sollutionChallenge/assets/apple_lr_1.0.0_beta.tflite", options=options)
if __name__ == '__main__':
    app.run(host="localhost",port=8800,debug=True,threaded=True)