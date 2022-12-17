import cv2

detectors = {
    "vz123": {
        "code": "vz123",
        "color": (50, 200, 50),
        "name": "Arbeitsstelle",
        "text": "Arbeitsstelle"
    },
    "vz205": {
        "code": "vz205",
        "color": (50, 50, 200),
        "name": "Vorfahrt gewähren!",
        "text": "Vorfahrt gewaehren"
    },
    "vz206": {
        "code": "vz206",
        "color": (50, 50, 200),
        "name": "Halt! Vorfahrt gewähren!",
        "text": "Stopp"
    },
    "vz267": {
        "code": "vz267",
        "color": (200, 50, 150),
        "name": "Verbot der Einfahrt",
        "text": "Verbot der Einfahrt"
    },
    "vz306": {
        "code": "vz306",
        "color": (200, 50, 50),
        "name": "Vorfahrtstraße",
        "text": "Vorfahrtstrasse"
    },
    "vz350": {
        "code": "vz350",
        "color": (50, 150, 250),
        "name": "Fußgängerüberweg",
        "text": "Fusgaengerueberweg"
    }
}

capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

for code, detector in detectors.items():
    detector["cascade"] = cv2.CascadeClassifier(f"detectors/{code}.detector.xml")

while True:
    _, image = capture.read()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for detector in detectors.values():
        faces = detector["cascade"].detectMultiScale(image_gray)

        for x, y, width, height in faces:
            cv2.rectangle(image, (x, y), (x + width, y + height), color = detector["color"], thickness = 1)
            cv2.putText(image, detector["text"], (x, y - 5), font, 0.5, detector["color"], 1)

    cv2.imshow("Camera", image)

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
