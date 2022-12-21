from turtle import circle
import cv2
import numpy as np

def erkennung(wert):
    for detector in detectors.values():
        if detector["form"] == wert:
            faces = detector["cascade"].detectMultiScale(image_gray)

            for x, y, width, height in faces:
                cv2.rectangle(image, (x, y), (x + width, y + height), color = detector["color"], thickness = 2)
                cv2.putText(image, detector["text"], (x, y - 5), font, 0.5, detector["color"], 2)

detectors = {
    "vz123": {
        "code": "vz123",
        "color": (50, 200, 50),
        "name": "Arbeitsstelle",
        "text": "Arbeitsstelle",
        "form": "triangle"
    },
    "vz205": {
        "code": "vz205",
        "color": (50, 50, 200),
        "name": "Vorfahrt gewähren!",
        "text": "Vorfahrt gewaehren",
        "form": "triangle"
    },
    "vz206": {
        "code": "vz206",
        "color": (50, 50, 200),
        "name": "Halt! Vorfahrt gewähren!",
        "text": "Stopp",
        "form": "octagon"
    },
    "vz267": {
        "code": "vz267",
        "color": (200, 50, 150),
        "name": "Verbot der Einfahrt",
        "text": "Verbot der Einfahrt",
        "form": "circle"
    },
    "vz306": {
        "code": "vz306",
        "color": (200, 50, 50),
        "name": "Vorfahrtstraße",
        "text": "Vorfahrtstrasse",
        "form": "square"
    },
    "vz350": {
        "code": "vz350",
        "color": (50, 150, 250),
        "name": "Fußgängerüberweg",
        "text": "Fusgaengerueberweg",
        "form": "square"
    }
}

capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

#Einlesen XML
for code, detector in detectors.items():
    detector["cascade"] = cv2.CascadeClassifier(f"detectors/{code}.detector.xml")

while True:
    _, image = capture.read()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Forms
    # Lines
    ret, thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Circle
    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 1.2, 100)

#Erkennung
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True), True)

        # Triangle
        if len(approx) == 3:
            erkennung("triangle")

        # Square & rectangle
        elif len(approx) == 4:
            (x,y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if ar >= 0.95 and ar <= 1.05:
                erkennung("square")
            else:
                erkennung("rectangle")

        # Octagon
        elif len(approx) == 8:
            erkennung("octagon")

    # Circle
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            erkennung("circle")

    cv2.imshow("Camera", image)

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
