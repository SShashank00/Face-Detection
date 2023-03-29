# Write a python file that captures images from your webcam video stream
# Extracts all faces from the image frame (using haarscades)
# Stores the face information into numpy arrays 


# 1. Read and show video stream ,capture images
# 2.Detect faces and show bounding box
# 3.Flatten the largest face image and save into numpy array
# 4.Repeat aboove for multiple people to genrate training data


import cv2
import numpy as np

# Int Camera
cap=cv2.VideoCapture(0)
# face detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
skip=0
face_data=[]
dataset_path='./data/'
file_name=input("Enter the name")
# branch_name=input("Enter the brach with year")

while True:
    ret ,frame=cap.read()
    gray_frame =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret==False:
        continue
    
    # cv2.imshow("GRAY Frame",gray_frame)


    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces, key = lambda f:f[2]*f[3])

    # Pick the largest face


    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
 
        # Region of intrest

        offset =10
        face_section = frame[y-offset: y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))

    
    cv2.imshow("Frame",frame)
    # cv2.imshow("face_section",face_section)

    # story every ten face
    if(skip%10==0):
        pass


    # compared 32 bit number and convert into 8 bit number 
    #  check from ascii value of q.

    key_pressed=cv2.waitKey(1)&0xFF
    if key_pressed== ord('q'):
        break

face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+file_name+'.npy',face_data)
print("Data Sucessfully save as "+dataset_path+file_name,'.npy')

cap.release()
cv2.destroyAllWindows()