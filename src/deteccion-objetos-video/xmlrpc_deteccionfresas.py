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
        
# Cargar la cámara con los parámetros intrínsecos y extrínsecos
def loadCamera():
    global myCamera
    myCamera = PinholeCamera()
    thetaY = 66 * DEGTORAD
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
    T = np.array([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, -252)])
    Res = np.dot(R_tot, T)
    RT = np.append(Res, [[0, 0, 0, 1]], axis=0)
    K = np.array([(FX, 0, CX, 0), (0, FY, CY, 0), (0, 0, 1, 0)])
    myCamera.rt = RT.copy()
    print(f"[DEBUG] Matriz RT de la cámara:\n{myCamera.rt}")


    # Asignar los parámetros a la cámara
    myCamera.position = np.array([0, 0, -252, 1])
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
    trans = myCamera.rt[:3, 3] # Extraer la traslación

    # Proyección inversa a coordenadas del mundo
    p3d_h = np.dot(inv_K, p2d_h)
    p3d_h = np.dot(inv_RT, p3d_h) 

    # Normalizar para obtener las coordenadas finales
    p3d[:2] = p3d_h[:2] / p3d_h[2]
    if p3d_h[2] != 0:
    	p3d = p3d_h / np.abs(p3d_h[2])  # Evitar errores de profundidad negativa
    else:
        p3d = p3d_h  #Mantener los valores originales si Z=0
    return p3d

# Obtener la intersección en Z para convertir las coordenadas 2D en 3D
def getIntersectionZ(p2d):
    # Proyección inversa a coordenadas 3D desde el punto 2D detectado
    p2d_h = np.array([p2d[0], p2d[1], 1])
    inv_K = np.linalg.inv(myCamera.k[:, :3])
    inv_RT = np.linalg.inv(myCamera.rt[:3, :3])

    # Calcular coordenadas en el espacio de la cámara
    p3d_h = np.dot(inv_K, p2d_h)
    p3d_h = np.dot(inv_RT, p3d_h)
    
    
    # Si el punto está en el suelo, Z sigue siendo 0, pero si es en altura, se debe calcular
    if np.abs(p3d_h[2]) > 1e-6: #Evita problemas con valores muy cercanos a 0
    	escala = -myCamera.position[2] / p3d_h[2]
    	p3d_h *= escala
    else:
        p3d_h *= 1 #Mantener la eslcala actual si Z es casi 0
        
    p3d = np.array(p3d_h)
    
    return p3d


# Calcular la distancia en 3D entre dos puntos
def calcular_distancia_3d(x_cam, y_cam, z_cam, x_punto, y_punto, z_punto):
    distancia = np.sqrt((x_punto - x_cam)**2 + (y_punto - y_cam)**2 + (z_punto - z_cam)**2)
    
    return distancia

print("[INFO] Iniciando detección de fresas en el frame actual...")

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
    

    # Configuración del dispositivo (CPU o GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    # Cargar los pesos del modelo
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device('cpu')))

    model.eval()  # Configurar el modelo en modo de evaluación
    classes = load_classes(opt.class_path)  # Cargar las clases
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Inicializar la captura de video (webcam o video pregrabado)
    if opt.webcam == 1:
        cap = cv2.VideoCapture(2)
    else:
        cap = cv2.VideoCapture(opt.directorio_video)

    loadCamera()  # Cargar los parámetros de la cámara
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")  # Colores para las clases

    previous_positions = []  # Para almacenar posiciones previamente detectadas
    threshold_distance = 5  # Aumentar margen para considerar que una detección sigue siendo la misma

    # Conexión con el robot usando XML-RPC
    server = SimpleXMLRPCServer(("0.0.0.0", 50000))
    server.RequestHandlerClass.protocol_version = "HTTP/1.1"
    print("Servidor XML-RPC corriendo en el puerto 50000...")

    # Función para obtener la siguiente posición
    def get_next_pose():
        if detected_points:
            pose = detected_points[-1]
            print(f"[DEBUG] Enviando última posición detectada: {pose}")
            return [pose[0],pose[1],pose[2],0,0,0]  # Devuelve el último punto detectado
        else:
            print(f"[DEBUG] No se detectaron puntos, enviando posición de inicio")
            return [0, 0, 0, 0, 0, 0]  # Devuelve una posición por defecto (sin movimiento)

    server.register_function(get_next_pose, "get_next_pose")


    # Ejecutar el servidor en un hilo aparte
    import threading
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

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
                
                for i, det in enumerate(detection):
                                     
                    # Solo continuar si la clase detectada es una fresa (ajustar según tus clases)
                    if len(det) == 7:  
                        x1, y1, x2, y2, cls_conf, conf, cls_pred = det

                                                                  
                        # Calcular el centro de la caja delimitadora
                        box_w = x2 - x1
                        box_h = y2 - y1
                        center_x = x1 + box_w // 2
                        center_y = y1 + box_h // 2

                        # Dibujar el recuadro de detección en el frame
                        color = (255, 0, 0)  # Azul para la caja delimitadora
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"Fresa - P{i+1}" #i es el índice de la detección
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        # Mostrar confianza de la detección correctamente
                        confidence_text = f"{float(cls_conf):.2f}"  # Asegurar que se muestra correctamente
                        cv2.putText(frame, confidence_text, (int(x2) + 10, int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Calcular las coordenadas 3D utilizando la función getIntersectionZ (similar al pinhole.py)
                        p2d = np.array([center_x, center_y])
                        pixelOnGround3D = getIntersectionZ(p2d)

                        # Extraer las coordenadas 3D del punto y convertirlas a tipo float
                        x_punto = float(pixelOnGround3D[0])
                        y_punto = float(pixelOnGround3D[1])
                        z_punto = float(pixelOnGround3D[2])

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
            print(f"Punto P{len(filtered_positions)} - Coordenadas 3D: X={x:.2f}, Y={y:.2f}, Z= 252.00")
            distancia = calcular_distancia_3d(0, 0, 0, x, y, 252)
            print(f"Punto P{len(filtered_positions)} - Distancia al punto: {distancia:.2f} milímetros")

            # Enviar la posición al robot usando XML-RPC
            try:
                detected_points.append((x, y, z))
            except Exception as e:
                print(f"[ERROR] No se pudo enviar la posición al robot: {e}")
            except xmlrpc.client.Fault as e:
                print(f"[ERROR] No se pudo enviar la posición al robot (Fault): {e}")
            except xmlrpc.client.ProtocolError as e:
                print(f"[ERROR] Error de protocolo al comunicarse con el robot: {e}")
            except xmlrpc.client.ResponseError as e:
                print(f"[ERROR] Error en la respuesta del servidor XML-RPC: {e}")
            except Exception as e:
                print(f"[ERROR] No se pudo enviar la posición al robot: {e}")
                print(f"[INFO] Posición enviada al robot: X={x_punto:.2f}, Y={y_punto:.2f}, Z={z_punto:.2f}")
            except Exception as e:
                print(f"[ERROR] No se pudo enviar la posición al robot: {e}")

        # Mostrar el frame con las detecciones
        cv2.imshow('Deteccion de Fresas', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

