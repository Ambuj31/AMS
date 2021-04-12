from keras.models import load_model
import cv2
import numpy as np
model = load_model('Ambuj.h5')

face_clsfr=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

labels_dict={0:'without_mask',1:'with_mask'}
color_dict={0:(0,255,0),1:(0,0,255)}

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for x,y,w,h in faces:
    
        face_img=gray[y:y+w,x:x+w]
        print(face_img)
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)
        print(result)
        
        label=np.argmax(result,axis=1)[0]
        print("label",label)
        if label==0:
             cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
             cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
           
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        #label=np.argmax(result,axis=1)[0]
      
       # cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
       # cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
       # cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()