import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import datetime
model=load_model("C:/Users/Lenovo/model-010.h5")

results={0:'without mask',1:'mask'}
GR_dict={0:(0,0,255),1:(0,255,0)}

rect_size = 4
cap = cv2.VideoCapture(0) 


haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print(haarcascade)
'''while True:
    """(rval, im) = cap.read()
    im=cv2.flip(im,1,1) 

    
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)"""
    
    ret, img = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray, 1.3, 5)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        
        face_img = img[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(300,300))
        normalized=rerect_sized/255.0
    
        reshaped=np.reshape(normalized,(1,300,300,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)

        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[label],-1)
        cv2.putText(img, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow('LIVE',   img)
    key = cv2.waitKey(10)
    
    if key == 27: 
        break
'''
while cap.isOpened():
  
    ret, img = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray, 1.1, 4, minSize=(30,30),flags = cv2.CASCADE_SCALE_IMAGE)
   # face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
    for(x,y,w,h) in faces:
        face_img = cv2.rectangle(img, (x, y), (x+w, y+h), (10, 159, 255), 2)
       # face_img = img[y:y+h, x:x+w]
        
        #cv2.imwrite('temp.jpg',face_img)
       # test_image=image.load_img('temp.jpg',target_size=(150,150,3))
       # test_image=image.img_to_array(test_image)
       # test_image=np.expand_dims(test_image,axis=0)
        pred=model.predict(face_img)
        if pred==1:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        datet=str(datetime.datetime.now())
        cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
          
    cv2.imshow('img',img)
    
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()

cv2.destroyAllWindows()