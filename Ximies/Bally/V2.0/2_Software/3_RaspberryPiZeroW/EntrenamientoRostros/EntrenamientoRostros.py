#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
- Proyecto: Bally 2.0
- Nombre del desarrollador: "Fernando Jurado Cote"
- Para: Xpikuos
- Version: 20.07.16.21.30.03
- Descripción: Entrena segun las personas de las que ha obtenido rostros, con las distintas emociones.
	guarda los resultados en dos modelos diferentes para poder posteriormente usar una u otra predicción
- https://creativecommons.org/licenses/by-nc-sa/4.0/deed.es
'''
import time
import numpy as np
import cv2 #sudo pip3 install opencv-contrib-python==4.1.0.25
import imutils
import os
import glob
import fnmatch

path = './Data/' #ruta donde están las imagenes a procesar

peopleList = os.listdir(path) #obtenemos la lista de personas y mostramos sus nombres
print('Lista de personas: ', peopleList)
#creamos los arrays para almacenar las etiquetas (nombre de la persona) 
# Y la información sobre las caras 
labels = []
facesData = []
label = 0
emociones = ["Alegre","Sorprendido","Triste","Enfadado"]
labelsemocion = []
facesDataemocion = []
labelemocion = 0

for nameDir in peopleList:
	personPath = path + nameDir
	print('Leyendo las imágenes')

	for fileName in os.listdir(personPath):
		#print('Rostros: ', nameDir  + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
	
		for i in range(0,len(emociones)):
			if (fileName.find(emociones[i]) != -1):
				#print(personPath +'/'+ fileName)
				gray = cv2.cvtColor(cv2.imread(personPath + '/' + fileName),cv2.COLOR_BGR2GRAY)
				facesDataemocion.append(gray)
				print(fileName)
				#print(fnmatch.filter(os.listdir(personPath),('*' + emociones[i] + '*')))
				labelsemocion.append(i)
			#tmp = str(fnmatch.filter(os.listdir(personPath),('*' + emociones[i] + '*')))
			#facesDataemocion.append(cv2.imread(personPath + '/'+ tmp,0))
			#labelemocion = labelemocion + 1
		#image = cv2.imread(personPath+'/'+fileName,0)
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
	label = label + 1
# He optado por LBPHFaceRecognizer por ser mas rapido en cargar y en reconocer los datos
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
print(labelsemocion)
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))
emotion_recognizer.train(facesDataemocion, np.array(labelsemocion))
# Almacenando el modelo obtenido
emotion_recognizer.write('emocionesLBPHFace.xml')
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")
