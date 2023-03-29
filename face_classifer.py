# Recognise Face using some classification algorithm - like Logistic KNN SVM etc;
"""1.Load the training data (numpy arrays of all the persons)
        # x-values are stored int the numpy arrays
        # y-values we need to assign for each person
2.Read a video stream using opencv
3.extract faces out of it
4. use KNN to find the prediction of face
5.map the predicted id to the name of the user
6.Display the predictins on the screen - bounding box and name"""
import cv2
import numpy as np
import os

#######################   KNN CODE     ############################################

def distance(v1,v2):  
    return np.sqrt(((v1-v2)**2).sum())

def knn(train ,test ,k = 5):
    dist=[]

    for i in range(train.shape[0]):
        # get the vector or label
        ix=train[i,:-1]
        iy=train[i,-1]
        # compute the disance from test point
        d=distance(test,ix)
        dist.append([d,iy])

    # sort based ondistance from test point
    dk = sorted(dist,key=lambda x: x[0])[:k]
    # Retrive only the labels
    labels =np.array(dk)[:,-1]

    # Get frequencies of each label
    output =np.unique(labels,return_counts=True)
    # Find maxfrequency and corresponding  label
    index =np.argmax(output[1])
    return output[0][index]

####################################################################################

# Int Camera
cap=cv2.VideoCapture(0)
# face detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

skip = 0
dataset_path='./data/'

face_data=[]

labels=[]
class_id=0  # labelsfor the given file
names={}    # Mappin btw id -name

# data preparation

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # creating a mapping id btw class and label name
        names[class_id] =fx[:-4]
    
        print("Loaded"+fx)
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)

        # create labels for the class
        target =class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset= np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)


while True:
    ret,frame=cap.read()
    if ret==False:
        continue

  
    faces=face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h=face

        offset =1
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        # Predict Label (out)
        out=knn(trainset,face_section.flatten())
        # Display on the screen the name and rectangle on it 
        pred_name =names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("faces",frame)

    key_pressed=cv2.waitKey(1)&0xFF
    if key_pressed== ord('q'):
       break

cap.release()
cv2.destroyAllWindows()





    
