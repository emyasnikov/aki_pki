import cv2

capture = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier(r"vorfahrtdetector.xml")
cascade2 = cv2.CascadeClassifier(r"arbeitsstelledetector.xml")
cascade3 = cv2.CascadeClassifier(r"verbotdereinfahrtdetector.xml")

while True:
    _, im = capture.read()
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    font=cv2.FONT_HERSHEY_SIMPLEX
    faces = cascade.detectMultiScale(im_gray)
    faces2 = cascade2.detectMultiScale(im_gray)
    faces3 = cascade3.detectMultiScale(im_gray)
    for x, y, width, height in faces:
        cv2.rectangle(im, (x,y), (x + width,y + height), color=(0, 140, 255), thickness=3)
        cv2.putText(im,'Vorfahrt',(x,y),font,0.9,(0,140,255),2)

    for x, y, width, height in faces2:
        cv2.rectangle(im, (x,y), (x + width,y + height), color=(0, 215, 255), thickness=3)
        cv2.putText(im,'Arbeitsstelle',(x,y),font,0.9,(0,215,255),2)

    for x, y, width, height in faces3:
        cv2.rectangle(im, (x,y), (x + width,y + height), color=(0, 0, 255), thickness=3)
        cv2.putText(im,'Verbot der Einfahrt',(x,y),font,0.9,(0,0,255),2)

    cv2.imshow("Kamera", im)
    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()