import cv2
import  matplotlib.pyplot as plt

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue

    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    cv2.imshow("Video Frame",frame)
    # cv2.imshow("Gray Frame",gray_frame)

    for (x,y,w,h)in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
    cv2.imshow("Video Frame",frame)


    key_pressed=cv2.waitKey(10)&0xFF

    if key_pressed==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
