###################################################################################
#                                                                                 #
#        Coordenadas del mundo real a partir de coordenadas de imagen usando      #
#              Una sola cámara calibrada basada en geometría analítica            #
#                                                                                 #
###################################################################################   

# Joko Siswantoro 1,2, Anton Satria Prabuwono 1, and Azizi Abdullah 1
#   
# 1 Center For Artificial Intelligence Technology, Faculty of Information Science and
#   Technology, Universiti Kebangsaan Malaysia, 43600 UKM, Bangi, Selangor D.E., Malaysia
#   
# 2 Department of Mathematics and Sciences, Universitas Surabaya,
#   Jl. Kali Rungkut Tengilis, Surabaya, 60293, Indonesia


import  yaml
import numpy as np
import cv2 as cv

# ---------------------------------------------------------------------------------
# Paso 1. Calibrar la cámara para obtener los parámetros extrínsecos R, T y
# parámetros intrínsecos de la cámara fx, fy, cx, cy (hecho en calibrateCamera.py)

# obtener la matriz de la cámara desde el archivo yaml
with open('./LogitechC310.yaml', 'r') as stream:
    try:
        calib_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# extraer la matriz de la cámara y los coeficientes de distorsión del archivo yaml
camera_matrix = np.array(calib_params['mtx']).reshape(3,3)

# cálculo de la media de las distancias focales (en píxeles)
fx = camera_matrix[0,0]
fy = camera_matrix[1,1]
mean_focul_length = (fx + fy) / 2

# Longitud del foco en mm como se especifica en la ficha técnica del producto
focul_length_in_mm = 4.4

# Calculo de pixeles por mm
pixles_per_mm = round(mean_focul_length / focul_length_in_mm)
# print('Pixles per mm: ', pixles_per_mm)

# Asunción de que los píxeles por mm son los mismos para x e y
sx = sy = pixles_per_mm

# centros ópticos
cx = camera_matrix[0,2]
cy = camera_matrix[1,2]

# ---------------------------------------------------------------------------------
# Paso 2. Encontrar la coordenada del punto p (xim, yim) en el sistema de coordenadas de la imagen.
# hecho en la parte de detección de objetos; usar puntos ficticios por ahora.

# obtener un conjunto de puntos ficticios
# image_points = np.array([[ 173.54767, 91.62902 ],
# [ 210.7686, 92.32755 ],
# [ 247.47551, 92.77607 ],
# [ 284.47098, 93.51474 ],
# [ 321.1669, 94.06741 ],
# [ 357.76126, 95.03036 ],
# [ 395.10233, 95.69413 ],
# [ 431.89203, 96.42651 ],
# [ 469.4813, 97.2708 ]])

image_points = np.array([[256,256], [300, 300],[150, 200], [50, 450]])

# convertir los puntos de la imagen en coordenadas homogéneas
image_points_hom = np.concatenate((image_points, np.ones((image_points.shape[0], 1))), axis=1)
# print("Homogeneous image points: \n", image_points_hom)

# ---------------------------------------------------------------------------------
# Paso 3. Hallar la coordenada del centro de proyección Oc en el sistema de coordenadas 
# del mundo real utilizando la Ec. (3).

# obtener los vectores de rotación y traslación correspondientes al campo de visión estático de la cámara
rotation_vector = np.array([2.10602313, 2.15303455, -0.19642816]).reshape(3,1)
translation_vector = np.array([-122.22738712, -117.38511046, 656.31453705]).reshape(3,1)

# encontrar el centro de la cámara Cc en el sistema de coordenadas del mundo real.
## obtener la matriz de rotación a partir del vector de rotación
rotation_matrix, _ = cv.Rodrigues(rotation_vector)

## centro de la cámara C = -R^(-1).t
C = - np.linalg.inv(rotation_matrix) @ translation_vector
# print("Camera Centre in world coordinate: \n", C)


for point in image_points_hom:
    print("Image point: ", np.round(point, decimals=2))

    # ---------------------------------------------------------------------------------
    # Paso 4. Encontrar la coordenada de p (ximw, yimw, zimw) en el mundo real 
    # utilizando la Ec. (7)

    # de las ec. 4, 5, 6
    x_imc = (point[0] - cx)/sx
    y_imc = (point[1] - cy)/sy
    z_imc = 1* focul_length_in_mm
    
    point_in_camera_coordinate = np.array([x_imc, y_imc, z_imc]).reshape(3,1)
    # print("Point in camera coordinate: \n", point_in_camera_coordinate)

    # de la ec. 7
    point_in_world_coordinate = np.linalg.inv(rotation_matrix) @ (point_in_camera_coordinate - translation_vector)
    # print("Point in world coordinate: \n", point_in_world_coordinate)

    # ecuación del plano donde se encuentra el punto en el sistema de coordenadas del mundo real
    # ecuación 10: Plano Z = 0. Almacenarlo en una matriz en forma estándar ax+by+cz = d
    plane_equation = np.array([0, 0, 1, 0]) # a, b, c, d
    
    # Calculo del parametro t usando la ec. 11 
    t = (plane_equation[-1] - np.dot(plane_equation[0:3], C)) / np.dot(plane_equation[0:3], (point_in_world_coordinate - C))
    # print("t: ", t)

    # ---------------------------------------------------------------------------------
    # Paso 5. Aproximar la coordenada de P en el sistema de coordenadas del mundo real utilizando las Ec.
    # (12), (13) y (14).

    # de las ec. 12, 13, 14
    approximated_world_coord = C + t * (point_in_world_coordinate - C)
    print("Approx: world coord: ", np.round(approximated_world_coord.reshape(3,), decimals=2))
    print("\n", '-'*50)
