import cv2
import numpy as np

print("Punti laterali del viso:\n")

#file xml di faccia, occhi per machine learning
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_lateral_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#inizializza la webcam
cap = cv2.VideoCapture(0)

while True:
        ret, img = cap.read()  #variabili dove si salva 'cosa' vede la webcam
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converte, attraverso la funzione built-in di cv2, l'immagine
                                                    #in bianco e nero, per ottimizzare i tempi (colori = più pesanti)
                                                        
        faces = face_cascade.detectMultiScale(gray,1.3,5) #localizza la faccia                            
        face_lateral = face_lateral_cascade.detectMultiScale(gray,1.3,5)
        smile = smile_cascade.detectMultiScale(gray,1.3,5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  #disegna un rettangolo (colore blue) per identificare la faccia
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = img[y:y+h,x:x+w]
            #sopra,denota i ROI

            #Stessa cosa, più difficile però, con gli occhi
            eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (sx,sy,sw,sh) in smile:
        
                    # cv2.rectangle(img,(sx,sy),(sx+sw,sy+sh),(255,0,0),2)
                    
            for (ex,ey,ew,eh) in eyes:                    
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),2)
                    
        for (lx,ly,lw,lh) in face_lateral:
                print(lx,ly,lw,lh)
                
        cv2.imshow("Face Detection", img)   #inizializza la finstra

        #tasto per uscire
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break



cap.release()
cv2.destroyAllWindows()
            
    


