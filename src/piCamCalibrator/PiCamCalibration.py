###############################################################################
# (C) 2019 - Julio Vega
###############################################################################
# Este programa coge las 20 imagenes de tablero de ajedrez tomadas por la     #
# Picam previamente, y que se encuentran en la carpeta "chess_board".         #
# En estas imagenes, el algoritmo encuentra las esquinas y finalmente devuelve#
# la matriz de parametros intrinsecos de la PiCam.                            #
###############################################################################

import cv2
import numpy as np
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 16x22 chess board, prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
object_point = np.zeros((16*22, 3), np.float32)
object_point[:, :2] = np.mgrid[0:22, 0:16].T.reshape(-1, 2)

# 3d point in real world space
object_points = []
# 2d points in image plane
image_points = []
h, w = 0, 0

images = glob.glob('chess_board/*.png')

for file_name in images:
    image = cv2.imread(file_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # find chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (22, 16), None)

    # add object points, image points
    if ret:
        object_points.append(object_point)
        cv2.cornerSubPix(gray, corners, (20, 20), (-1, -1), criteria)
        image_points.append(corners)

        # draw and display the corners
        cv2.drawChessboardCorners(image, (22, 16), corners, ret)
        cv2.imshow('image', image)
        cv2.waitKey(500)

if len(object_points) > 0 and len(image_points) > 0:

    # calibration
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (w, h), None, None)

    print ("camera matrix:\n", cameraMatrix)
    print ("=====================================")
    print ("=====================================")
    # pi camera intrinsic parameters
    fx = cameraMatrix[0, 0] # distancia focal f_x
    fy = cameraMatrix[1, 1] # distancia focal f_y
    u0 = cameraMatrix[0, 2] # centro optico c_x
    v0 = cameraMatrix[1, 2] # centro optico c_y

    print ("Parametros interesantes de la matriz:")
    print ("=====================================")
    print ("Distancia focal [Fx, Fy] =", "[", fx, ", ", fy, "]")
    print ("Centro optico [Cx, Cy] o [u0, v0] =", "[", u0, ", ", v0, "]\n")
    print ("Coeficientes de distorsion =", distCoeffs)
    
else:
    print ("No se encontraron esquinas en ninguna de las imágenes. La calibración no es posible.")
cv2.destroyAllWindows()
