from eval import *
import mxnet as mx
import cv2
from collections import namedtuple
from os.path import join
from mxnet import nd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import sys

model_prefix = 'models/My_FER_model'
num_epoch = 1000
mod, Batch = build_model(model_prefix, num_epoch)
video = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video.read()

    # Predict result with network
    img = format_image(frame)
    if img is not None:

        mod.forward(Batch([nd.array(img)]))
        result = mod.get_outputs()[0].asnumpy()
        feelings_faces = get_emojis(EMOTIONS)
        draw_bars(frame, result, feelings_faces)
    else:
        #cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 6, (255, 0, 0), 20)
        cv2.imshow('FER', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video.release()
cv2.destroyAllWindows()