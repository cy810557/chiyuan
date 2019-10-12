import dlib
import cv2
import numpy as np
from imutils import face_utils

class FaceModel:
    def __init__(self, model_name=None):
        self.detector = dlib.get_frontal_face_detector()
        if model_name is not None:
            self.predictor = dlib.shape_predictor(model_name)
        else: 
            self.predictor = None

    def predict(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        for (i, rect) in enumerate(rects):
            box = [rect.left(), rect.top(), rect.right(), rect.bottom()]
            cv2.rectangle(
              image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
            )
            if self.predictor is not None:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        return image

if __name__ == "__main__":
    model = FaceModel("models/shape_predictor_5_face_landmarks.dat")
    img = cv2.imread("images/face1.jpg")
    rst = model.predict(img)
    cv2.imshow("predict", rst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


