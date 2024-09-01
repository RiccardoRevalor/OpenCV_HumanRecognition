import cv2
import numpy as np

humanity = 0
a = 0
faceok = flatok = eyeok = smileok = False
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_lateral_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()  #variabili dove si salva 'cosa' vede la webcam
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5) #localizza la faccia                            
    face_lateral = face_lateral_cascade.detectMultiScale(gray,1.3,5)
    

    for (x,y,w,h) in faces:
        if (x,y,w,h) in faces:
            humanity +=1
            faceok = True
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = img[y:y+h,x:x+w]
            if a >=12: cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
    if faceok == True:
        for (lx,ly,lw,lh) in face_lateral:
            if (lx,ly,lw,lh) in face_lateral:                
                 humanity +=1
                 flatok = True
            if flatok == True:
                for (ex,ey,ew,eh) in eyes:
                    if (ex,ey,ew,eh) in eyes:
                        humanity += 1
                        eyeok = True
                        print("Localizzato soggetto umano")
                        a +=1
                        if a >=12:        
                            cv2.imshow("Face Detection", img)   #inizializza la finstra
                            #tasto per uscire
                            k = cv2.waitKey(30) & 0xff
                            if k == 27:
                                break
cap.release()
cv2.destroyAllWindows()
            
                        
            
        
        
        
