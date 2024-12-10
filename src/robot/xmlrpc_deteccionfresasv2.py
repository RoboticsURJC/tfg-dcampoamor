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

print("[INFO] Iniciando programa...")

# Modelo Pinhole de cámara
class PinholeCamera:
    def __init__(self):
        self.position = np.zeros(4)
        self.k = np.zeros((3, 4))
        self.rt = np.zeros((4, 4))
        
# Definir la función getIntersectionZ para proyectar puntos del plano de imagen al espacio 3D        
def getIntersectionZ(p2d):
    # Convertir coordenadas de píxeles a coordenadas homogéneas
    p2d_h = np.array([p2d[0], p2d[1], 1])

    # Obtener la matriz intrínseca y la matriz de rotación y traslación de la cámara
    K = myCamera.k[:, :3]
    RT = myCamera.rt[:3, :3]

    # Calcular la proyección inversa para obtener las coordenadas 3D del punto en el suelo (Z = 0)
    inv_K = np.linalg.inv(K)
    inv_RT = np.linalg.inv(RT)

    p3d_h = np.dot(inv_K, p2d_h)
    p3d_h = np.dot(inv_RT, p3d_h)

    # Normalizar para obtener las coordenadas en el espacio 3D
    p3d = p3d_h / p3d_h[2]

    # Devolver las coordenadas 3D del punto
    return p3d


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
   
    print("[INFO] Parámetros de la cámara cargados correctamente.")

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

# Configuración del servidor XML-RPC

server = xmlrpc.server.SimpleXMLRPCServer(("localhost", 50000), allow_none=True, bind_and_activate=False)
try:
    server.server_bind()
    server.server_activate()
except OSError as e:
    if e.errno == 98:  # Address already in use
        print("[ERROR] The address is already in use. Please use another port or stop the process occupying the port.")

# Registrar introspección y funciones
server.register_introspection_functions()


# Definir la función para enviar la posición al robot
def send_position_to_robot(positions):
    # Enviar coordenadas como una lista de tuplas
    if positions:
        print(f"Fresa detectada en: {positions}")
    return True

server.register_function(send_position_to_robot, "send_position_to_robot")

print("[INFO] Iniciando detección de fresas en el frame actual...")

def get_detected_points():
    return detected_points

# Registrar la función para obtener los puntos detectados
server.register_function(get_detected_points, "get_detected_points")

if __name__ == "__main__":
    # Argumentos de línea de comandos para la configuración del modelo y otros parámetros
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
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
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")  # Colores para las clases

    previous_positions = []  # Para almacenar posiciones previamente detectadas
    threshold_distance = 0.1  # Aumentar margen para considerar que una detección sigue siendo la misma

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

        # Extraer las coordenadas de las fresas detectadas
        positions = []
        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, opt.img_size, frame.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    cls_conf = conf  # Usar directamente el valor de confianza
                    # Solo continuar si la clase detectada es una fresa (ajustar según tus clases)
                    if int(cls_pred) == 0:  # Suponiendo que la clase 0 es "fresa"
                        # Calcular el centro de la caja delimitadora
                        box_w = x2 - x1
                        box_h = y2 - y1
                        center_x = x1 + box_w // 2
                        center_y = y1 + box_h // 2

                        # Dibujar el recuadro de detección en el frame
                        color = (255, 0, 0)  # Azul para la caja delimitadora
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, "Fresa", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        cv2.putText(frame, f"{cls_conf:.2f}", (int(x2) + 10, int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Calcular las coordenadas 3D utilizando la función getIntersectionZ (similar al pinhole.py)
                        p2d = np.array([center_x, center_y])
                        pixelOnGround3D = getIntersectionZ(p2d)

                        # Extraer las coordenadas 3D del punto
                        x_punto = pixelOnGround3D[0]
                        y_punto = pixelOnGround3D[1]
                        z_punto = pixelOnGround3D[2]

                        # Añadir las coordenadas 3D a las posiciones detectadas
                        positions.append((x_punto, y_punto, z_punto))
                        
                        # Obtener las coordenadas de la cámara
                        x_cam = myCamera.position[0]
                        y_cam = myCamera.position[1]
                        z_cam = myCamera.position[2]
                        
                        # Calcular la distancia desde la cámara al punto detectado
                        distancia = calcular_distancia_3d(x_cam, y_cam, z_cam, x_punto, y_punto, z_punto)

        # Filtrar detecciones similares basadas en la distancia
        filtered_positions = []
        for pos in positions:
            is_new_detection = True
            for prev_pos in previous_positions:
                dist = calcular_distancia_3d(prev_pos[0], prev_pos[1], prev_pos[2], pos[0], pos[1], pos[2])
                if dist < threshold_distance:
                    is_new_detection = False
                    break
            if is_new_detection:
                filtered_positions.append(pos)
                previous_positions.append(pos)

        # Mostrar las coordenadas 3D y la distancia por terminal si es una nueva detección
        for (x, y, z) in filtered_positions:
            print(f"Punto P{len(positions)} - Coordenadas 3D: X={x_punto:.2f}, Y={y_punto:.2f}, Z={z_punto:.2f}")
            print(f"Punto P{len(positions)} - Distancia al punto: {distancia:.2f} milímetros")

        # Mostrar el frame con las detecciones
        cv2.imshow('Deteccion de Fresas', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

