import cv2
import numpy as np
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer
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

# Variable Global para almacenar los puntos detectados a enviar por XML-RPC
detected_points = []
stop_server = False  # Variable global para detener el servidor

print("[INFO] Iniciando programa...")

# Modelo Pinhole de cámara
class PinholeCamera:
    def __init__(self):
        self.position = np.zeros(4)
        self.k = np.zeros((3, 4))
        self.rt = np.zeros((4, 4))
        
def loadCamera():
    global myCamera
    myCamera = PinholeCamera()
    thetaY = 0 * DEGTORAD
    thetaZ = 0 * DEGTORAD
    thetaX = 0 * DEGTORAD

    R_y = np.array([(np.cos(thetaY), 0, -np.sin(thetaY)), 
                    (0, 1, 0),
                    (np.sin(thetaY), 0,  np.cos(thetaY))])
    R_z = np.array([(np.cos(thetaZ), -np.sin(thetaZ), 0),
                    (np.sin(thetaZ),  np.cos(thetaZ), 0),
                    (0,               0,              1)])
    R_x = np.array([(1,              0,               0),
                    (0, np.cos(thetaX),  np.sin(thetaX)),
                    (0, -np.sin(thetaX), np.cos(thetaX))])

    R_subt = np.dot(R_y, R_z)
    R_tot = np.dot(R_subt, R_x)

    T = np.array([(1, 0, 0,   0),
                  (0, 1, 0,   0),
                  (0, 0, 1, -410)])
    Res = np.dot(R_tot, T)
    RT = np.append(Res, [[0, 0, 0, 1]], axis=0)
    K = np.array([(FX,   0, CX, 0),
                  (0,   FY, CY, 0),
                  (0,    0,  1, 0)])
    myCamera.rt = RT.copy()
    print(f"[DEBUG] Matriz RT de la cámara:\n{myCamera.rt}")

    myCamera.position = np.array([0, 0, -410, 1])
    myCamera.k = K
    myCamera.rt = RT
   
    print("[INFO] Parámetros de la cámara cargados correctamente.")

def pixel2optical(p2d):
    aux = p2d[0]
    p2d[0] = LARGO_IMAGEN - 1 - p2d[1]
    p2d[1] = aux
    return p2d

def backproject(p2d, camera):
    p3d = np.zeros(3)
    p2d = pixel2optical(p2d)
    p2d_h = np.array([p2d[0], p2d[1], 1])
    inv_K = np.linalg.inv(camera.k[:, :3])
    inv_RT = np.linalg.inv(camera.rt[:3, :3])
    p3d_h = np.dot(inv_K, p2d_h)
    p3d_h = np.dot(inv_RT, p3d_h) 
    if p3d_h[2] != 0:
        p3d = p3d_h / np.abs(p3d_h[2])
    else:
        p3d = p3d_h
    return p3d

def getIntersectionZ(p2d, invert_y=False):
    p2d_h = np.array([p2d[0], p2d[1], 1])
    inv_K = np.linalg.inv(myCamera.k[:, :3])
    inv_RT = np.linalg.inv(myCamera.rt[:3, :3])
    p3d_h = np.dot(inv_K, p2d_h)
    p3d_h = np.dot(inv_RT, p3d_h)
    if np.abs(p3d_h[2]) > 1e-6:
        escala = -myCamera.position[2] / p3d_h[2]
        p3d_h *= escala
    # Para depurar, opcionalmente invertir la componente Y
    if invert_y:
        p3d_h[1] *= -1
    # Imprimir valores intermedios para comprobar
    # print(f"p2d_h: {p2d_h}, p3d_h: {p3d_h}")
    return np.array(p3d_h)


def calcular_distancia_3d(x_cam, y_cam, z_cam, x_punto, y_punto, z_punto):
    return np.sqrt((x_punto - x_cam)**2 + (y_punto - y_cam)**2 + (z_punto - z_cam)**2)

# Esta función compara dos listas de posiciones usando un umbral
def positions_are_similar(list1, list2, threshold):
    if len(list1) != len(list2):
        return False
    for pos in list1:
        found = False
        for pos2 in list2:
            if calcular_distancia_3d(pos[0], pos[1], pos[2], pos2[0], pos2[1], pos2[2]) < threshold:
                found = True
                break
        if not found:
            return False
    return True

print("[INFO] Iniciando detección de fresas en el frame actual...")

if __name__ == "__main__":
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device('cpu')))
    model.eval()
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    if opt.webcam == 1:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(opt.directorio_video)

    loadCamera()
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

    threshold_distance = 10  # Se mantiene para filtrar detecciones cercanas
    # Nuevo umbral para determinar si las posiciones son lo suficientemente similares (suavizado)
    stability_threshold = 10

    # Variable para almacenar las posiciones impresas del fotograma anterior
    last_printed_positions = []

    # Conexión con el robot usando XML-RPC
    server = SimpleXMLRPCServer(("192.168.23.107", 50000))
    server.RequestHandlerClass.protocol_version = "HTTP/1.1"
    print("Servidor XML-RPC corriendo en el puerto 50000...")

    def get_next_pose():
        if detected_points:
            pose = detected_points[-1]
            print(f"[DEBUG] Enviando última posición detectada: {pose}")
            return [pose[0], pose[1], pose[2], 0, 0, 0]
        else:
            print(f"[DEBUG] No se detectaron puntos, enviando posición de inicio")
            return [0, 0, 0, 0, 0, 0]

    server.register_function(get_next_pose, "get_next_pose")
    import threading
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    while cap:
        ret, frame = cap.read()
        if not ret:
            break

        RGBimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, opt.img_size)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))

        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        positions = []
        if detections:
            for detection in detections:
                if detection is not None:
                    detection = rescale_boxes(detection, opt.img_size, frame.shape[:2])
                    for i, det in enumerate(detection):
                        if len(det) == 7:  
                            x1, y1, x2, y2, cls_conf, conf, cls_pred = det
                            box_w = x2 - x1
                            box_h = y2 - y1
                            center_x = x1 + box_w // 2
                            center_y = y1 + box_h // 2
                            color = (255, 0, 0)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            label = f"Fresa - P{i+1}"
                            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            confidence_text = f"{float(cls_conf):.2f}"
                            cv2.putText(frame, confidence_text, (int(x2) + 10, int(y1)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            p2d = np.array([center_x, center_y])
                            pixelOnGround3D = getIntersectionZ(p2d)
                            x_punto = float(pixelOnGround3D[0])
                            y_punto = float(pixelOnGround3D[1])
                            z_punto = float(pixelOnGround3D[2])
                            positions.append((x_punto, y_punto, z_punto))

        # Filtrar detecciones similares solo dentro del fotograma actual
        filtered_positions = []
        for pos in positions:
            is_new_detection = True
            for fpos in filtered_positions:
                dist = calcular_distancia_3d(fpos[0], fpos[1], fpos[2],
                                             pos[0], pos[1], pos[2])
                if dist < threshold_distance:
                    is_new_detection = False
                    break
            if is_new_detection:
                filtered_positions.append(pos)

        # Solo se imprimen si las posiciones actuales difieren de las impresas anteriormente,
        # usando stability_threshold para considerar el ruido
        if not positions_are_similar(filtered_positions, last_printed_positions, stability_threshold):
            for idx, (x, y, z) in enumerate(filtered_positions, start=1):
                print(f"Punto P{idx} - Coordenadas 3D: X={x:.2f}, Y={y:.2f}, Z=410.00")
                distancia = calcular_distancia_3d(0, 0, 0, x, y, 343)
                print(f"Punto P{idx} - Distancia al punto: {distancia:.2f} milímetros")
                try:
                    detected_points.append((x, y, z))
                except Exception as e:
                    print(f"[ERROR] No se pudo enviar la posición al robot: {e}")
            last_printed_positions = filtered_positions.copy()

        cv2.imshow('Deteccion de Fresas', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

