#!/usr/bin/env python
# coding: utf-8

# # Proyecto
# ## Vision Computacional
# ## Emmanuel Antonio Cuevas
# 
# El siguiente proyecto presenta análisis de videos de fútbol. Para realizar esta tarea escogimos video panorámicos tomados desde dos camaras, para tener vista completa del campo y hacer más preciso nuestro análisis.

# #### Las librerías que usaremos 

# In[19]:


from __future__ import print_function
from scipy.optimize import minimize
from imutils.object_detection import non_max_suppression
from imutils import paths

import tensorflow as tf
import numpy as np
import argparse
import imutils
import random
import time
import math
import cv2
import sys


# #### Detector de Objetos
# Para esto usamos los modelos implementados y preentrenados de tensorflow. Nos apoyamos de la API de tensorflow, el siguiente código se encuentra la función para inicializar el detector y para procesar un frame

# In[2]:


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        #print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


# #### Dibujos
# El siguiente código es para dibujar circulos en una imágen, dado un vector de centros se dibujan sobre la imagen circulos de un radio especificado

# In[3]:


def drawCircles(image, centers, color = (0, 255, 0), alpha = 1, r = 5):
    numb, _ = centers.shape
    
    auximg = np.zeros(image.shape)
    np.copyto(auximg, image)
    auximg = auximg.astype(image.dtype)
    
    for i in range(numb):
        x = int(round(centers[i][0]))
        y = int(round(centers[i][1]))
        
        cv2.circle(auximg, (x, y), 5, color, -1)
        
    image = cv2.addWeighted(image, 1 -alpha, auximg, alpha, 0)
    return image


# #### Funciones para detectar clics
# Para hacer las mediciones necesitamos puntos de referencia en la imagen, estos son seleccionados manualmente en el area de la izquierda y luego en la de la derecha. 

# In[4]:


def onMouse1(event, x, y, flags, param):
   global goal1_dst
   if event == cv2.EVENT_LBUTTONDOWN:
        goal1_dst.append((x, y))
        cv2.circle(image_aux,(x,y),5,(0,0,255),-1)
        
def onMouse2(event, x, y, flags, param):
   global goal2_dst
   if event == cv2.EVENT_LBUTTONDOWN:
        goal2_dst.append((x, y))
        cv2.circle(image_aux,(x,y),5,(255,0,0),-1)


# #### Función de penalty
# Para estimar la longitud del campo tendremos diferentes medida, escogemos la longitud tal que minimice la distancia hacia todas las posibles distancias estimadas

# In[5]:


def euclideanPenalty(x):
    global distances
    
    sol = 0
    for i in range(3):
        sol += ( (x -distances[i])*(x -distances[i]) )
        
    return sol


# #### Distancia 
# Para calcular la distancia a la que se encuentran dos puntos en el "modelo real". Proyectamos de coordenas en la imagen a coordenadas en nuestro modelo y ahí calculamos la distancia Euclideana.

# In[44]:


# Pasamos del campo a un modelo 2D y ahí calculamos las distancias
def distance(x1, y1, x2, y2, invh):
    pts = [[x1, y1], [x2, y2]]
    pts = np.array([pts], dtype = float)
    planepts = cv2.perspectiveTransform(pts, invh)
    
    x1 = planepts[0][0][0]
    y1 = planepts[0][0][1]
    x2 = planepts[0][1][0]
    y2 = planepts[0][1][1]
    
    d1 = float(x1 -x2)
    d2 = float(y1 -y1)
    d = d1 * d1 + d2 * d2
    return math.sqrt(d) 


# #### Proyectamos una imagen al centro del campo

# In[7]:


def imageInField(image, name, alpha = 0.25):
    global fieldlen, fieldorg
    # Colocar imagen dentro del circulo central
    asset = cv2.imread(name, cv2.IMREAD_COLOR)
    
    # rediminezionamos para que la altura de la imagen 
    # este dentro del circulo central
    # Se ajusta el ancho de acuerdo al aspecto original de la imagen
    ratio = asset.shape[1] / asset.shape[0]
    asset = cv2.resize(asset, (int(round(183*ratio)), 183 ))

    # Creamos imagen a proyectar del tamaño del campo
    temp = np.zeros(fieldorg.shape, dtype = fieldorg.dtype)
    center = [int(round(fieldlen / 2.0)), 450]


    y1 = int(center[1] -asset.shape[0] / 2)
    y2 = y1 +asset.shape[0]

    x1 = int(center[0] -asset.shape[1] / 2)
    x2 = x1 +asset.shape[1]
    
    # Compiamos la imagen en el centro
    temp[y1:y2, x1:x2] = asset

    # Proyectamos
    temp = cv2.warpPerspective(temp, h, (image.shape[1], image.shape[0]))
    output = cv2.addWeighted(image, 1, temp, alpha, 0)
    
    return output


# #### Tracking
# 
# Usamos tracking para seguir a algun jugador y esimar su velocidad. Esta funcion inicializa una instancia de tracking

# In[ ]:


# Tracker
def newTracker():
    tracker = cv2.TrackerMIL_create()
    return tracker


# ## Comienza el código principal
# 
# Comenzamos abriendo el video y extrayendo el primer frame. De este seleccionamos los puntos que se mencionan en el reporte, primero del area de la izquierda y luego de la derecha.

# In[77]:


#vidcap = cv2.VideoCapture(sys.argv[1])
vidcap = cv2.VideoCapture("./videos/soccer.mp4")
succes, frame = vidcap.read()

width, height, _ = frame.shape
frame = cv2.resize(frame, (1900, 975))


image_aux = np.array(frame)
# Calculamos los puntos de homografía para la primera porteria (izquierda)
goal1_dst = []

cv2.namedWindow('selectCornersGoal1')
cv2.setMouseCallback('selectCornersGoal1', onMouse1)

while(len(goal1_dst) < 11):
    cv2.imshow('selectCornersGoal1', image_aux)
    cv2.waitKey(100)

cv2.destroyAllWindows()
cv2.waitKey(1)


# Calculamos los pnutos de homografía para la segunda porteria (derecha)
goal2_dst = []

cv2.namedWindow('selectCornersGoal2')
cv2.setMouseCallback('selectCornersGoal2', onMouse2)

while(len(goal2_dst) < 11):
    cv2.imshow('selectCornersGoal2', image_aux)
    cv2.waitKey(100)

cv2.destroyAllWindows()
cv2.waitKey(1)

goal1_dst = np.array(goal1_dst)
goal2_dst = np.array(goal2_dst)


# Se conocen algunas medidas que se usan en el fútbol, apartir de estas medidas estimamos la longitud del campo tomando la longitud que minimice la distancia hacia las demás

# In[78]:


distances = np.zeros(3)

distances[0] = 165.0 * (goal2_dst[0][0] -goal1_dst[0][0]) / (goal1_dst[8][0] -goal1_dst[0][0])
distances[1] = 55.0  * (goal2_dst[1][0] -goal1_dst[1][0]) / (goal1_dst[5][0] -goal1_dst[1][0])
distances[2] = 55.0  * (goal2_dst[4][0] -goal1_dst[4][0]) / (goal1_dst[6][0] -goal1_dst[4][0])

start = distances[0]
res = minimize(euclideanPenalty, start)

fieldlen = res.x[0]
fieldlen = int(round(fieldlen))


# In[79]:


print("Longitud estimada del campo: ", fieldlen)


# Creamos un sistema de coordenadas que corresponden a un modelo "real" del campo

# In[80]:


# Puntos en el plano
goal1_src = np.array( [ [0, 248.4],
                        [0, 358.4],
                        [0, 413.4],
                        [0, 486.6],
                        [0, 541.6],
                        [55.0, 358.4],
                        [55.0, 541.6],
                        [110.0, 450.0], 
                        [165.0, 248.4],
                        [165.0, 651.6],
                        [201.5, 450.0]
                        ] )

goal2_src = np.array( [ [fieldlen, 248.4],
                        [fieldlen, 358.4],
                        [fieldlen, 413.4],
                        [fieldlen, 486.6],
                        [fieldlen, 541.6],
                        [fieldlen -55.0, 358.4],
                        [fieldlen -55.0, 541.6],
                        [fieldlen -110.0, 450.0], 
                        [fieldlen -165.0, 248.4],
                        [fieldlen -165.0, 651.6],
                        [fieldlen -201.5, 450.0]
                        ] )


# Estimamos la matriz de homografía entre la imagen y nuestro sistema

# In[81]:


srcpts = []
dstpts = []

for i in range(11):
    srcpts.append( [goal1_src[i][0], goal1_src[i][1]] )
for i in range(11):
    srcpts.append( [goal2_src[i][0], goal2_src[i][1]] )

for i in range(11):
    dstpts.append( [goal1_dst[i][0], goal1_dst[i][1]] )
for i in range(11):
    dstpts.append( [goal2_dst[i][0], goal2_dst[i][1]] )
    
srcpts = np.array(srcpts)
dstpts = np.array(dstpts)

# Calculamos la matriz de homografia y la matriz para revertir
h, status = cv2.findHomography(srcpts, dstpts)
invh, status = cv2.findHomography(dstpts, srcpts)


# Proyectamos la imagen del modelo hacia el campo para ver que tan buena estimación es

# In[85]:


field = cv2.imread("./imagenes/field.jpg", cv2.IMREAD_COLOR)
fieldorg = cv2.resize(field, ( int(round(fieldlen)), field.shape[1] ))
field = np.array(fieldorg)

field = drawCircles(field, goal1_src, (255, 255, 255))
field = drawCircles(field, goal2_src, (255, 255, 255))

cv2.line(field, tuple((int(goal1_src[0][0]), int(goal1_src[0][1]))), tuple((int(goal1_src[8][0]), int(goal1_src[8][1]))), (255,255,255), 3)
cv2.line(field, tuple((int(goal1_src[8][0]), int(goal1_src[8][1]))), tuple((int(goal1_src[9][0]), int(goal1_src[9][1]))), (255,255,255), 3)
cv2.line(field, tuple((int(goal1_src[1][0]), int(goal1_src[1][1]))), tuple((int(goal1_src[5][0]), int(goal1_src[5][1]))), (255,255,255), 3)
cv2.line(field, tuple((int(goal1_src[5][0]), int(goal1_src[5][1]))), tuple((int(goal1_src[6][0]), int(goal1_src[6][1]))), (255,255,255), 3)
cv2.line(field, tuple((int(goal1_src[6][0]), int(goal1_src[6][1]))), tuple((int(goal1_src[4][0]), int(goal1_src[4][1]))), (255,255,255), 3)
cv2.line(field, tuple((int(0), int(651))), tuple((int(goal1_src[9][0]), int(goal1_src[9][1]))), (255,255,255), 3)

cv2.line(field, tuple((int(goal2_src[0][0]), int(goal2_src[0][1]))), tuple((int(goal2_src[8][0]), int(goal2_src[8][1]))), (255,255,255), 3)
cv2.line(field, tuple((int(goal2_src[8][0]), int(goal2_src[8][1]))), tuple((int(goal2_src[9][0]), int(goal2_src[9][1]))), (255,255,255), 3)
cv2.line(field, tuple((int(goal2_src[1][0]), int(goal2_src[1][1]))), tuple((int(goal2_src[5][0]), int(goal2_src[5][1]))), (255,255,255), 3)
cv2.line(field, tuple((int(goal2_src[5][0]), int(goal2_src[5][1]))), tuple((int(goal2_src[6][0]), int(goal2_src[6][1]))), (255,255,255), 3)
cv2.line(field, tuple((int(goal2_src[6][0]), int(goal2_src[6][1]))), tuple((int(goal2_src[4][0]), int(goal2_src[4][1]))), (255,255,255), 3)
cv2.line(field, tuple((int(fieldlen), int(651))), tuple((int(goal2_src[9][0]), int(goal2_src[9][1]))), (255,255,255), 3)


cv2.imshow("field", field)
cv2.waitKey(3000)

im_out = cv2.warpPerspective(field, h, (image_aux.shape[1], image_aux.shape[0]))

cv2.imshow("field", im_out)
cv2.waitKey(3000)

dst = cv2.addWeighted(image_aux, 0.7, im_out, 0.3,0)

cv2.imshow('field',dst)
cv2.waitKey(3000)
cv2.destroyAllWindows()
cv2.waitKey(1)


# 
# Inicializamos el mapa de calor con puros ceros

# In[86]:


heatMapI = np.zeros((field.shape[0], field.shape[1]), dtype = np.uint8)


# Cargamos el mapa de inferencia del detector de objetos preentrenado

# In[87]:


# Cargamos el modelo para detectar jugadores
model_path = './trained_soccer_model/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.02


# ### Empezamos a procesar frame por frame
# 
# Empieza la parte iterativa e interactiva. Los comandos que se pueden usar son
# 
# s - Muestra una imagen en el centro del campo
# 
# h - Muestra el mapa de calor del juego
# 
# v - Muestra la velocidad de algun jugador al azar
# 
# q - detener el video

# In[91]:


W, H = 1900, 975

showImage = 0
showMap = 0
disVel = 0


tracker  = cv2.TrackerMIL_create()

start = []
kalman = newKalman()

fps = vidcap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_video = cv2.VideoWriter('output.avi', fourcc, fps, (W, H))

duration = int(fps*5)
vel = np.array([0, 0, 0, 0, 0, 0, 0], dtype = float)

while True:
    
    boxes, scores, classes, num = odapi.processFrame(frame)

    players_pos = []
    
    aux_boxes = []
    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i]
            # Nos quedamos con las cajas que pertenecen a jugadores
            x = (box[1] + box[3])/2.0
            y = box[2]
            
            players_pos.append( [x, y] )
            aux_boxes.append(box)
            
    boxes = np.array(aux_boxes)
    
    
    
    # Procesamos las posiciones de los jugadores encontrados y agregamos calor en las zonas donde se encuentran
    players_pos = np.array([players_pos], dtype = float)
    plane_pos = cv2.perspectiveTransform(players_pos, invh)
    
    temp = np.zeros((field.shape[0], field.shape[1]), dtype = np.uint8)
    for i in range(plane_pos.shape[1]):
        x = int( round(plane_pos[0][i][0]) )
        y = int( round(plane_pos[0][i][1]) )
        
        cv2.circle(temp, (x, y), 10, 255, -1)
        
    heatMapI = cv2.addWeighted(heatMapI, 1, temp, 0.005, 0)
    heatMap = cv2.applyColorMap(heatMapI, cv2.COLORMAP_JET)
    
    
    
    # Procesamos la imagen que se mostrara
    if showImage > 0:
        # Se muestra una imagen en el centro del campo
        if showImage < 10:            
            alpha = showImage * 0.25 / 10.0
            frame = imageInField(frame, "./imagenes/UGTO.png", alpha)
        else:
            frame = imageInField(frame, "./imagenes/UGTO.png")
        showImage -= 1
        
    elif showMap > 0:
        # Se muestra el mapa de calor en el campo
        alpha = 0.3
        
        if showMap < 10:
            alpha = showMap * 0.3 / 10.0
            
        heatMap = cv2.warpPerspective(heatMap, h, (frame.shape[1], frame.shape[0]))
        frame = cv2.addWeighted(frame, 1, heatMap, alpha,0)
        showMap -= 1
        
    elif disVel > 0:
        # Se muestra la velocidad de un jugador al azar
        prevx = trackerbox[0] +trackerbox[2]/2.0
        prevy = trackerbox[1] +trackerbox[3]
        
        success, trackerbox = tracker.update(frame)
        
        
        if success:
            p1 = (int( round(trackerbox[0]) ), int( round(trackerbox[1]) ))
            p2 = (int( round(trackerbox[0]) + trackerbox[2]), int( round(trackerbox[1] + trackerbox[3]) ))
            
            posx = (p1[0] +p2[0])/2.0
            posy = p2[1]
            
            dist = distance(prevx, prevy, posx, posy, invh)
            
            # Se toma la mediana de las últimas aproximaciones de velocidad
            vel[0] = vel[1]
            vel[1] = vel[2]
            vel[2] = vel[3]
            vel[3] = vel[4]
            vel[4] = (dist * fps * 3.6) / 10.0 
            
            if (duration -disVel)%3 == 0:
                prevvel = currvel
                currvel = np.median(vel)
                currvel = round((currvel +prevvel) / 2.0, 2)
            
            # Se muestra la velocidad sobre el jugador
            font = cv2.FONT_HERSHEY_SIMPLEX
            TxtVel = "v: " + str(currvel) + "km/h"
            cv2.putText(frame, TxtVel, p1, font, 1,(255,255,255),2,cv2.LINE_AA)
            
        disVel -= 1
    
    
    
    # Se muestra el frame en pantalla y se almacena en un video
    output_video.write(frame)
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)
    
    
    
    # A continuacion se preprocesan las instrucciones que se hayan dado desde teclado
    # Operaciones validas desde teclado
    #        q - detener 
    #        s - mostrar imagen en el centro del campo (duracion 100 frames)
    #        h - mostrar mapa de calor 
    #        v - velocidad de un jugador 
    # No se muestra mas de uno a la vez
    
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        if showImage == 0 and showMap == 0 and disVel == 0:
            showImage = duration
    elif key & 0xFF == ord('h'):
        if showImage == 0 and showMap == 0 and disVel == 0:
            showMap = duration
    elif key & 0xFF == ord('v'):
        if showImage == 0 and showMap == 0 and disVel == 0:
            disVel = duration
            
            x = random.randint(0, boxes.shape[0])
            trackerbox = list( boxes[x] )
            
            trackerbox[2] = trackerbox[2] -trackerbox[0]
            trackerbox[3] = trackerbox[3] -trackerbox[1]
            
            trackerbox[0], trackerbox[1] = trackerbox[1], trackerbox[0]
            trackerbox[2], trackerbox[3] = trackerbox[3], trackerbox[2]
            
            trackerbox = tuple(trackerbox)
            
            tracker = newTracker()
            tracker.init(frame, trackerbox)
        
        
        
    # Leemos el siguiente frame en el video
    success, frame = vidcap.read()
    if not success:
        break
    
    # Redimenzionamos 
    frame = cv2.resize(frame, (W, H))
    
output_video.release()
cv2.destroyAllWindows()
cv2.waitKey(1)


# In[ ]:




