#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
- Proyecto: Bally 2.0
- Nombre del desarrollador: "Fernando Jurado Cote"
- Para: Xpikuos
- Version: 20.07.16.23.55.10
- Descripción: Reconoce rostros y emociones a partir de los parametros de la base de datos creada en los anteriores.
	Muestra nombre de la persona, coordenadas donde se encuentra y su estado de animo.
- https://creativecommons.org/licenses/by-nc-sa/4.0/deed.es
'''
import time
import numpy as np
import cv2 #sudo pip3 install opencv-contrib-python==4.1.0.25
import imutils
import os

path = './Data/'
emociones = ["Alegre","Sorprendido","Triste","Enfadado"]
imagePaths = os.listdir(path)
print('imagePaths=',imagePaths)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Leyendo el modelo
face_recognizer.read('modeloLBPHFace.xml')
emotion_recognizer.read('emocionesLBPHFace.xml')

cap = cv2.VideoCapture(0)
time.sleep(0.5) #damos Tiempo a la cámara para que arranque
# bajamos resolucion y frame rate
cap.set(3,320)
cap.set(4,240)
cap.set(cv2.CAP_PROP_FPS, 5)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
	ret,frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = gray.copy()

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
		emociondetectada = emotion_recognizer.predict(rostro)
		persona = face_recognizer.predict(rostro)
		#A partir de aquí descomentar todo si quereis ver en video los resultados
		#Están comentados para ahorrar recursos y trabajar sin escritorio

		#cv2.putText(frame,'{}'.format(emociondetectada),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
		
		if persona[1] < 70:
			print(imagePaths[persona[0]] + 'esta en x =', (x+(w/2)), ', y =', (y+(h/2)))
			if emociondetectada[1] < 70:
				print(imagePaths[persona[0]] + 'esta un poco ' + emociones[emociondetectada[0]])
			#cv2.putText(frame,'{}'.format(emociones[emociondetectada[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
 			#cv2.putText(frame,'{}'.format(imagePaths[persona[0]]),(x,y-25,),2,1.1,(0,255,0),1,cv2.LINE_AA)
			#cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		#else:
			#cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			#cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                
		
	#cv2.imshow('frame',frame)
    
	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
