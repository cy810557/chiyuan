from imutils.video import VideoStream
import cv2

vs = VideoStream(src=0).start()
while True:
    frame = vs.read()
    cv2.imshow("figure", frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
vs.stop()
