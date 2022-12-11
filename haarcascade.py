'''
https://www.youtube.com/watch?v=zwiKIzvGAhM
https://github.com/opencv/opencv/tree/master/data/haarcascades
'''

import cv2

capture = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("detectors/vz123.detector.xml")

while True:
    _, image = capture.read()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(image_gray)

    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color = (255, 0, 0), thickness = 5)

    cv2.imshow("Camera", image)

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
