import cv2
import numpy as np
import xmlrpc.server
import torch
import argparse
from models import Darknet
from utils.utils import load_classes, rescale_boxes, non_max_suppression
from utils.datasets import pad_to_square, resize
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable

# Parámetros de Cámara y proyección
ANCHO_IMAGEN = 640
LARGO_IMAGEN = 480
FX = 816.218
FY = 814.443
CX = 316.068
CY = 236.933
DEGTORAD = 3.1415926535897932 / 180

# Variables para la navegación con OpenGL
angle_x = 0
angle_y = 0
zoom = -5
camera_x = 0
camera_y = 0
camera_z = 0
mouse_last_x = 0
mouse_last_y = 0
is_dragging = False

# Variables Globales para almacenar los puntos detectados
detected_points = []
stop_server = False  # Nueva variable global para detener el servidor

# Modelo Pinhole de cámara
class PinholeCamera:
    def __init__(self):
        self.position = np.zeros(4)
        self.k = np.zeros((3, 4))
        self.rt = np.zeros((4, 4))

# Cargar la cámara con los parámetros intrínsecos y extrínsecos
def loadCamera():
    global myCamera
    myCamera = PinholeCamera()
    thetaY = 65 * DEGTORAD
    thetaZ = 0 * DEGTORAD
    thetaX = 0 * DEGTORAD

    # Rotación en los ejes Y, Z y X
    R_y = np.array([(np.cos(thetaY), 0, -np.sin(thetaY)), (0, 1, 0), (np.sin(thetaY), 0, np.cos(thetaY))])
    R_z = np.array([(np.cos(thetaZ), -np.sin(thetaZ), 0), (np.sin(thetaZ), np.cos(thetaZ), 0), (0, 0, 1)])
    R_x = np.array([(1, 0, 0), (0, np.cos(thetaX), np.sin(thetaX)), (0, -np.sin(thetaX), np.cos(thetaX))])

    # Combinación de las rotaciones
    R_subt = np.dot(R_y, R_z)
    R_tot = np.dot(R_subt, R_x)

    # Transformación de la cámara
    T = np.array([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, -265)])
    Res = np.dot(R_tot, T)
    RT = np.append(Res, [[0, 0, 0, 1]], axis=0)
    K = np.array([(FX, 0, CX, 0), (0, FY, CY, 0), (0, 0, 1, 0)])

    # Asignar los parámetros a la cámara
    myCamera.position = np.array([0, 0, -265, 1])
    myCamera.k = K
    myCamera.rt = RT

# Convertir las coordenadas de píxeles a coordenadas ópticas
def pixel2optical(p2d):
    aux = p2d[0]
    p2d[0] = LARGO_IMAGEN - 1 - p2d[1]
    p2d[1] = aux
    return p2d

# Proyectar hacia atrás las coordenadas 2D a 3D
def backproject(p2d, camera):
    p3d = np.zeros(3)
    p2d = pixel2optical(p2d)
    p2d_h = np.array([p2d[0], p2d[1], 1])

    # Inversión de la matriz intrínseca y la matriz de rotación-traslación
    inv_K = np.linalg.inv(camera.k[:, :3])
    inv_RT = np.linalg.inv(camera.rt[:3, :3])

    # Proyección inversa a coordenadas del mundo
    p3d_h = np.dot(inv_K, p2d_h)
    p3d_h = np.dot(inv_RT, p3d_h)

    # Normalizar para obtener las coordenadas finales
    p3d[:2] = p3d_h[:2] / p3d_h[2]
    p3d[2] = 0
    return p3d

# Obtener la intersección en Z para convertir las coordenadas 2D en 3D
def getIntersectionZ(p2d):
    p3d = backproject(p2d, myCamera)
    return p3d

# Calcular la distancia en 3D entre dos puntos
def calcular_distancia_3d(x_cam, y_cam, z_cam, x_punto, y_punto, z_punto):
    distancia = np.sqrt((x_punto - x_cam)**2 + (y_punto - y_cam)**2 + (z_punto - z_cam)**2)
    return distancia

# Configurar el servidor XMLRPC
server = xmlrpc.server.SimpleXMLRPCServer(("localhost", 8000))
print("Servidor XML-RPC corriendo en el puerto 8000...")

# Definir la función para enviar la posición al robot
def send_position_to_robot(positions):
    # Enviar coordenadas como una lista de tuplas
    if positions:
        print(f"Enviando coordenadas al robot: {positions}")
    return True

server.register_function(send_position_to_robot, "send_position_to_robot")

if __name__ == "__main__":
    # Argumentos de línea de comandos para la configuración del modelo y otros parámetros
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--webcam", type=int, default=1, help="Is the video processed from webcam? 1 = Yes, 0 = No")
    parser.add_argument("--directorio_video", type=str, help="Directorio al video")
    opt = parser.parse_args()
    print(opt)

    # Configuración del dispositivo (CPU o GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    # Cargar los pesos del modelo
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Configurar el modelo en modo de evaluación
    classes = load_classes(opt.class_path)  # Cargar las clases
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Inicializar la captura de video (webcam o video pregrabado)
    if opt.webcam == 1:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(opt.directorio_video)

    loadCamera()  # Cargar los parámetros de la cámara
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    previous_positions = []  # Para almacenar posiciones previamente detectadas
    threshold_distance = 0.3  # Margen para considerar que una detección sigue siendo la misma
    detected_objects = []  # Lista para almacenar detecciones activas

    while cap:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen de BGR a RGB para el modelo
        RGBimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, opt.img_size)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))

        # Realizar la detección con el modelo
        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        positions = []
        current_detected_objects = []
        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    # Calcular el centro de la caja delimitadora
                    box_w = x2 - x1
                    box_h = y2 - y1
                    center_x = x1 + box_w // 2
                    center_y = y1 + box_h // 2
                    p2d = np.array([center_x, center_y])
                    p3d = getIntersectionZ(p2d)
                    X, Y, Z = p3d[0], p3d[1], p3d[2]
                    current_position = (X, Y, Z)

                    # Filtrar detecciones similares basadas en la distancia
                    is_new_detection = True
                    for prev_pos in previous_positions:
                        dist = calcular_distancia_3d(prev_pos[0], prev_pos[1], prev_pos[2], X, Y, Z)
                        if dist < threshold_distance:
                            is_new_detection = False
                            break

                    # Añadir siempre la detección a la lista de detecciones actuales
                    current_detected_objects.append((x1, y1, x2, y2, conf, cls_pred))

                    # Si es una nueva detección, añadir a la lista de posiciones
                    if is_new_detection:
                        positions.append(current_position)
                        previous_positions.append(current_position)

        # Dibujar todas las detecciones, incluso si no son nuevas
        for (x1, y1, x2, y2, conf, cls_pred) in current_detected_objects:
            color = [int(c) for c in colors[int(cls_pred)]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Borde más grueso
            cv2.putText(frame, "Fresa", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Texto "Fresa" en la esquina superior izquierda
            cv2.putText(frame, f"{conf:.2f}", (x2 - 50, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Porcentaje de confianza en la esquina superior derecha

        # Enviar las posiciones detectadas al robot
        if positions:
            send_position_to_robot(positions)

        # Mostrar el frame con las detecciones
        cv2.imshow('Deteccion de Fresas', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
