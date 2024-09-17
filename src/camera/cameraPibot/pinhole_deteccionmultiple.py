###############################################################################
# (C) 2019 - Julio Vega
###############################################################################
# Algoritmo de navegación de PiBot basado en sonar visual percibido por PiCam.#
# En este programa, los motores están parados, para así poder hacer pruebas.  #
###############################################################################

# -*- coding: utf-8 -*-
from projGeom import *
import cv2
import numpy as np

# defino variables globales:
ANCHO_IMAGEN = 640
LARGO_IMAGEN = 480
FX = 816.218
FY = 814.443
CX = 316.068
CY = 236.933

DEGTORAD = 3.1415926535897932 / 180
myCamera = None

def loadCamera():
    global myCamera
    myCamera = PinholeCamera()
    thetaY = 65*DEGTORAD
    thetaZ = 0*DEGTORAD
    thetaX = 0*DEGTORAD

    R_y = np.array([(np.cos(thetaY),0,-np.sin(thetaY)),(0,1,0),(np.sin(thetaY),0,np.cos(thetaY))])
    R_z = np.array([(np.cos(thetaZ),-np.sin(thetaZ),0),(np.sin(thetaZ),np.cos(thetaZ),0),(0,0,1)])
    R_x = np.array([(1,0,0),(0,np.cos(thetaX),np.sin(thetaX)),(0, -np.sin(thetaX),np.cos(thetaX))])

    R_subt = np.dot(R_y, R_z)
    R_tot = np.dot(R_subt, R_x)

    T = np.array([(1,0,0,0),(0,1,0,0),(0,0,1,-265)])
    Res = np.dot(R_tot, T)
    RT = np.append(Res, [[0,0,0,1]], axis=0)
    K = np.array([(FX,0,CX,0),(0,FY,CY,0),(0,0,1,0)])

    myCamera.position.x = 0
    myCamera.position.y = 0
    myCamera.position.z = -265
    myCamera.position.h = 1

    myCamera.k11 = K[0,0]
    myCamera.k12 = K[0,1]
    myCamera.k13 = K[0,2]
    myCamera.k14 = K[0,3]

    myCamera.k21 = K[1,0]
    myCamera.k22 = K[1,1]
    myCamera.k23 = K[1,2]
    myCamera.k24 = K[1,3]

    myCamera.k31 = K[2,0]
    myCamera.k32 = K[2,1]
    myCamera.k33 = K[2,2]
    myCamera.k34 = K[2,3]

    myCamera.rt11 = RT[0,0]
    myCamera.rt12 = RT[0,1]
    myCamera.rt13 = RT[0,2]
    myCamera.rt14 = RT[0,3]

    myCamera.rt21 = RT[1,0]
    myCamera.rt22 = RT[1,1]
    myCamera.rt23 = RT[1,2]
    myCamera.rt24 = RT[1,3]

    myCamera.rt31 = RT[2,0]
    myCamera.rt32 = RT[2,1]
    myCamera.rt33 = RT[2,2]
    myCamera.rt34 = RT[2,3]

    myCamera.rt41 = RT[3,0]
    myCamera.rt42 = RT[3,1]
    myCamera.rt43 = RT[3,2]
    myCamera.rt44 = RT[3,3]

    myCamera.fdistx = K[0,0]
    myCamera.fdisty = K[1,1]
    myCamera.u0 = K[0,2]
    myCamera.v0 = K[1,2]
    myCamera.rows = LARGO_IMAGEN
    myCamera.columns = ANCHO_IMAGEN

def pixel2optical(p2d):
    aux = p2d.x
    p2d.x = LARGO_IMAGEN-1-p2d.y
    p2d.y = aux
    p2d.h = 1
    return p2d

def getIntersectionZ(p2d):
    p3d = Punto3D()
    res = Punto3D()
    p2d_ = Punto2D()

    x = myCamera.position.x
    y = myCamera.position.y
    z = myCamera.position.z

    p2d_ = pixel2optical(p2d)
    result, p3d = backproject(p2d_, myCamera)

    if((p3d.z-z) == 0.0):
        res.h = 0.0
        return

    zfinal = 0.

    xfinal = x + (p3d.x - x)*(zfinal - z)/(p3d.z-z)
    yfinal = y + (p3d.y - y)*(zfinal - z)/(p3d.z-z)    

    res.x = xfinal
    res.y = yfinal
    res.z = zfinal
    res.h = 1.0

    return res

def calcular_distancia_3d(x_cam, y_cam, z_cam, x_punto, y_punto, z_punto):
    distancia = np.sqrt((x_punto - x_cam)**2 + (y_punto - y_cam)**2 + (z_punto - z_cam)**2)
    return distancia

def detect_color(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Aplicar operaciones de morfología para reducir el ruido
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    for contour in contours:
        # Obtener el centroide del contorno
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
    return centroids

def getPoints(frame):
    global fronteraImg
    global puntosFrontera

    pixel = Punto2D()
    pixelOnGround3D = Punto3D()

    lower_color = np.array([35, 100, 100])
    upper_color = np.array([85, 255, 255])
    
    centroids = detect_color(frame, lower_color, upper_color)

    for idx, centroid in enumerate(centroids):
        centroid_x = int(centroid[0])
        centroid_y = int(centroid[1])
        
        cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"P{idx+1} ({centroid_x},{centroid_y})", (centroid_x - 20, centroid_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        pixel.x = centroid_x
        pixel.y = centroid_y
        pixel.h = 1

        pixelOnGround3D = getIntersectionZ(pixel)

        x_punto = pixelOnGround3D.x
        y_punto = pixelOnGround3D.y
        z_punto = pixelOnGround3D.z

        x_cam = myCamera.position.x
        y_cam = myCamera.position.y
        z_cam = myCamera.position.z

        distancia = calcular_distancia_3d(x_cam, y_cam, z_cam, x_punto, y_punto, z_punto)

        print(f"Punto P{idx+1} - Coordenadas 3D: X={x_punto}, Y={y_punto}, Z={z_punto}")
        print(f"Punto P{idx+1} - Distancia al punto: {distancia:.2f} milímetros")

if __name__=="__main__":
    cv2.namedWindow("Image Feed")
    cv2.moveWindow("Image Feed", 159, -25)

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        exit()
    
    loadCamera()
    
    while True:
        ret, frame = cap.read() 
        if not ret:
            print("Error: No se puede capturar el frame")
            break
        
        flipped_frame = cv2.flip(frame, 1)
        getPoints(flipped_frame)
        
        cv2.imshow("Image Feed", flipped_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

