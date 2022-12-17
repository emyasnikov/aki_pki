'''
https://www.youtube.com/watch?v=zwiKIzvGAhM
https://github.com/opencv/opencv/tree/master/data/haarcascades
'''

import cv2

detectors = {
    "vz123": "Arbeitsstelle",
    "vz205": "Vorfahrt gewähren!",
    "vz206": "Halt! Vorfahrt gewähren!",
    "vz267": "Verbot der Einfahrt",
    "vz306": "Vorfahrtstraße",
    "vz350": "Fußgängerüberweg"
}

capture = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("detectors/vz123.detector.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    _, image = capture.read()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(image_gray)

    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color = (255, 0, 0), thickness = 5)
        cv2.putText(image, "Test", (x, y), font, 0.9, (255, 0, 0), 2)

    cv2.imshow("Camera", image)

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
