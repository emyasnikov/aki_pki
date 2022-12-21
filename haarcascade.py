from turtle import circle
import cv2
import numpy as np
import time

def detection(value):
    for detector in detectors.values():
        if detector["form"] == value:
            faces = detector["cascade"].detectMultiScale(image_gray)

            for x, y, width, height in faces:
                cv2.rectangle(image, (x, y), (x + width, y + height), color = detector["color"], thickness = 2)
                cv2.putText(image, detector["text"], (x, y - 5), font, 0.5, detector["color"], 2)

def timeInMS():
    return time.time() * 1000

detectors = {
    "vz123": {
        "code": "vz123",
        "color": (50, 200, 50),
        "form": "triangle",
        "name": "Arbeitsstelle",
        "text": "Arbeitsstelle"
    },
    "vz205": {
        "code": "vz205",
        "color": (50, 50, 200),
        "form": "triangle",
        "name": "Vorfahrt gewähren!",
        "text": "Vorfahrt gewaehren"
    },
    "vz206": {
        "code": "vz206",
        "color": (50, 50, 200),
        "form": "octagon",
        "name": "Halt! Vorfahrt gewähren!",
        "text": "Stopp"
    },
    "vz267": {
        "code": "vz267",
        "color": (200, 50, 150),
        "form": "circle",
        "name": "Verbot der Einfahrt",
        "text": "Verbot der Einfahrt"
    },
    "vz306": {
        "code": "vz306",
        "color": (200, 50, 50),
        "form": "square",
        "name": "Vorfahrtstraße",
        "text": "Vorfahrtstrasse"
    },
    "vz350": {
        "code": "vz350",
        "color": (50, 150, 250),
        "form": "square",
        "name": "Fußgängerüberweg",
        "text": "Fusgaengerueberweg"
    }
}

capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
last_time = timeInMS()

# Read XML
for code, detector in detectors.items():
    detector["cascade"] = cv2.CascadeClassifier(f"detectors/{code}.detector.xml")

while True:
    _, image = capture.read()

    # Run only at 30 frames per second
    if timeInMS() > last_time + 33:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Lines
        ret, thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Circle
        circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1.2, 100)

        # Detection
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True), True)

            # Triangle
            if len(approx) == 3:
                detection("triangle")

            # Square & rectangle
            elif len(approx) == 4:
                (x,y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                if ar >= 0.95 and ar <= 1.05:
                    detection("square")
                else:
                    detection("rectangle")

            # Octagon
            elif len(approx) == 8:
                detection("octagon")

        # Circle
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detection("circle")

    last_time = timeInMS()

    cv2.imshow("Camera", image)

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
