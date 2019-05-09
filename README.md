# SoccerAnalisys
Computer Vision Class -Project

El programa no recibe parámetros.

Al ejecutar se abrira una ventana mostrando el primer frame, en esta ventana se seleccionaron los puntos del área de la izquierda en el orden

	A11, A12, A13, ..., A19, A110, A111
	
como se menciona el reporte y en la imagen /imagenes/orden.png

Luego se abrirá otra ventana para seleccionar los puntos del área de la derecha.

El código es interactivo, se deberán precionar las teclas s, h, v o q para ejecutar alguna de las características: 

	imagen en el centro del campo
	
	mapa de calor
	
	velocidad de un jugador al azar
	
	salir del programa

El video se guardara en un archivo .avi llamado output.avi.

Se recomienda usar tensorflow con GPU pues el tiempo del detector puede ser muy tardado (aprox 1s por frame usando solo CPU).

Librerías a usar

tensorflow

numpy

argparse

imutils

random

scipy

time

math

cv2

sys

Si omití información escribanme a
	emmanuel.antonio@cimat.mx
