from dlib_detector.face_model import FaceModel
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
from datetime import datetime
import imutils
import time
import cv2

# initialize the putput frame and a lock used to ensure thread-safe excahges of the output frames
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to warmup

# for RPi camera:
# vs = VideoStream(usePiCamera=1).start()

# for usb camera:
vs = VideoStream(src=0).start()
time.sleep(2.0)

# next function will render template file: index.html and serve up the output video stream:
@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def detect_landmarks(frameCount):
    # grab global references to the video stream, output frame, 
    # and lock variables
    global vs, outputFrame, lock

    # TODO: initialize landmark model
    # model = FaceModel()
    model = FaceModel("dlib_detector/models/shape_predictor_68_face_landmarks.dat")

    # loop over frames from the video stream
    while True:
        # read the next frame, resize it, convert to grayscale and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        # TODO: apply related algorithm to captured frame.
        pred = model.predict(frame)


        # grab the current timestamp and draw it on the frame
        timestamp = datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S:%p"), (10, frame.shape[0]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)

        # TODO: visualize detection results
        
        # acquire the lock, set the output frame, and release the lock
        # We need to acquire the lock to ensure the outputFrame variable 
        # is not accidentally being read by a client while we are trying 
        # to update it.
        with lock:
            outputFrame = frame.copy()

def generate():
        # grab global references to the output frame and lock variables
        global outputFrame, lock

        while True:
            # check if the output frame is avaliable, otherwise skip
            if outputFrame is None:
                continue

            # encode the frame in JPEG format (compress raw image)
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
            
            # yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media 
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    args.add_argument("-p", "--port", type=int, required=True, 
        help="ephemeral port number of the server (1024 to 65535)")
    args.add_argument("-f", "--frame_count", type=int, default=32, 
        help="# of frames used to constructh the background model")
    #TODO: algorithm related args

    args = vars(args.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_landmarks, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True, 
        threaded=True, use_reloader=False)

vs.stop()


