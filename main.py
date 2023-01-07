from colors import colors
from detectors import detectors
from turtle import circle
import cv2
import numpy as np
import time
import argparse
import os

#Parameter initialisieren
parser = argparse.ArgumentParser( prog = 'Haarcascade Verkehrszeichen', description = 'Erkennung und Klassifizierung von Straßenschildern mit Haarcascades', epilog = '')
parser.add_argument('-d', '--directory', default='', help='Relativer Pfad zu den Eingabebildern. Wenn nicht angegeben, dann wird Video gestartet.' )

def detection(values, image):
    for value in values:
        for detector in detectors.values():
            if detector["form"] == value:
                faces = detector["cascade"].detectMultiScale(image_gray)

                for x, y, width, height in faces:
                    if width > 50:
                        cv2.rectangle(image, (x, y), (x + width, y + height), color = colors[detector["color"]], thickness = 2)
                        cv2.putText(image, detector["text"], (x, y - 5), font, 0.5, colors[detector["color"]], 2)

def timeInMS():
    return time.time() * 1000

# Parameter einlesen
args = parser.parse_args()

# camera / Directory initialisieren
bWithDir = True     # Flag ob Camera oder Directory Mode
capture = None      # Objekt für Zugriff auf Kamera
images = []         # Array mit den Pfaden zu den einzelnen Bildern (Directory Mode)
currentImg = 0      # Index des aktuell angezeigten Bildes (Directory Mode)

if args.directory == '':
    bWithDir = False
    capture = cv2.VideoCapture(0)
else:
    for filename in os.listdir(args.directory):
        f = os.path.join(args.directory, filename)
        if os.path.isfile(f):
            images.append(f)

# font initialisieren
font = cv2.FONT_HERSHEY_SIMPLEX

# time für Framerate initialisieren
last_time = timeInMS()

#Einlesen XML
for code, detector in detectors.items():
    detector["cascade"] = cv2.CascadeClassifier(f"detectors/{code}.detector.xml")

while True:
    # Bild einlesen bzw. von der Kamera abgreifen (try/except, weil openCV mit äöüß im Dateinamen nicht klar kommt)
    try:
        if not bWithDir:
            _, image = capture.read()
        else:
            image = cv2.imread( images[currentImg] )
    except:
        break

    # Run only at 30 frames per second
    if timeInMS() > last_time + 33:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Lines
        ret, thresh = cv2.threshold(image_gray, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Circle
        circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT,1.5,100, param1=300,param2=80,minRadius=1,maxRadius=0)

        contour_set = set()
        # Detection Form
        for contour in contours:
               x,y,w,h = cv2.boundingRect(contour)
               if w > 10:
                approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True), True)

                # Triangle
                if len(approx) == 3:
                    contour_set.add("triangle")

                # Square & rectangle
                elif len(approx) == 4:
                    (x,y, w, h) = cv2.boundingRect(approx)
                    ar = w / float(h)
                    if ar >= 0.95 and ar <= 1.05:
                        contour_set.add("square")
                    else:
                        contour_set.add("rectangle")

                # Octagon
                elif len(approx) == 8:
                    contour_set.add("octagon")

        # Circle
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                contour_set.add("circle")

        delta = timeInMS() - last_time
        last_time = timeInMS()
        fps = str(int(1000/delta))

        cv2.putText(image, fps, (10, 20), font, 0.5, (255, 255, 255), 1)

    detection(contour_set, image)

    cv2.imshow("Camera", image)

    # Tasteneingabe abgreifen
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("n"):
        currentImg += 1
        if currentImg >= len(images):
            currentImg = 0

# Ressourcen freigeben
if not bWithDir:
    capture.release()
cv2.destroyAllWindows()
