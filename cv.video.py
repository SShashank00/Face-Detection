import cv2

# capture the image 
cap=cv2.VideoCapture(0)
while True:
    # ret is boolean value either 0 or 1
    ret,frame=cap.read()
    # convert color
    gray_frame =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret==False:
        continue
    
    cv2.imshow("Video Frame",frame)
    cv2.imshow("GRAY Frame",gray_frame)

    # compared 32 bit number and convert into 8 bit number 
    #  check from ascii value of q.

    key_pressed=cv2.waitKey(1)&0xFF
    if key_pressed== ord('q'):
        break


cap.release()
cv2.destroyAllWindows()