#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer
import torch
import argparse
import threading
from models import Darknet
from utils.utils import load_classes, rescale_boxes, non_max_suppression
from utils.datasets import pad_to_square, resize
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable

# Importar librerías para OpenGL y GLFW
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

# Parámetros de cámara y proyección (para detección y visualización)
ANCHO_IMAGEN = 640
LARGO_IMAGEN = 480
FX = 816.218
FY = 814.443
CX = 316.068
CY = 236.933
DEGTORAD = 3.1415926535897932 / 180

# Variables globales
detected_points = []  # Para enviar posiciones al robot vía XMLRPC
stop_server = False

print("[INFO] Iniciando programa...")

# ---------------------- CÓDIGO DE PROYECCIÓN CON OPENGL ---------------------- #
def ray_plane_intersection(ray_origin, ray_direction, plane_point, plane_normal):
    """
    Calcula la intersección de un rayo con un plano.
    """
    denom = np.dot(ray_direction, plane_normal)
    if np.abs(denom) < 1e-6:
        return None  # Rayo paralelo al plano.
    t = np.dot(plane_point - ray_origin, plane_normal) / denom
    if t < 0:
        return None  # Intersección detrás del origen.
    return ray_origin + t * ray_direction

class CameraProjection:
    def __init__(self, width=800, height=600):
        if not glfw.init():
            raise Exception("No se pudo inicializar GLFW")
        self.width = width
        self.height = height
        self.window = glfw.create_window(width, height, "Proyección de Cámara", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("No se pudo crear la ventana GLFW")
        glfw.make_context_current(self.window)
        # Registrar callbacks para scroll y ratón (botón y movimiento)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        self.init_gl()
        
        # Parámetros de la cámara virtual y del plano imagen
        self.camera_position = np.array([0.0, 0.0, 0.0])
        self.image_plane_z = 1.0
        self.image_plane_width = 2.0
        self.image_plane_height = 1.5
        
        # Definición del plano "pared"
        self.wall_plane_point = np.array([0.0, 0.0, 5.0])
        self.wall_plane_normal = np.array([0.0, 0.0, 1.0])
        
        # Rectángulo de detección (coordenadas del plano imagen)
        self.det_rect = {
            "xmin": -self.image_plane_width / 4,
            "xmax":  self.image_plane_width / 4,
            "ymin": -self.image_plane_height / 4,
            "ymax":  self.image_plane_height / 4,
        }
        
        # Variables para navegación 3D
        self.yaw = 0        # ángulo horizontal en grados
        self.pitch = 0      # ángulo vertical en grados
        self.distance = 7.0 # distancia desde el target
        
        # Variables para manejo del ratón
        self.is_dragging = False
        self.last_cursor_x = 0
        self.last_cursor_y = 0

    def init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
    
    def scroll_callback(self, window, xoffset, yoffset):
        # yoffset es positivo al desplazar hacia arriba y negativo al bajar
        self.distance -= yoffset * 0.5  # Factor de zoom ajustable
        if self.distance < 1:
            self.distance = 1

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            self.is_dragging = True
            self.last_cursor_x, self.last_cursor_y = glfw.get_cursor_pos(window)
        elif button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            self.is_dragging = False

    def cursor_pos_callback(self, window, xpos, ypos):
        if self.is_dragging:
            dx = xpos - self.last_cursor_x
            dy = ypos - self.last_cursor_y
            # Ajusta la sensibilidad del movimiento (aquí 0.1)
            self.yaw += dx * 0.1
            self.pitch -= dy * 0.1
            if self.pitch > 89:
                self.pitch = 89
            if self.pitch < -89:
                self.pitch = -89
            self.last_cursor_x = xpos
            self.last_cursor_y = ypos

    def process_input(self):
        # Puedes mantener controles de teclado opcionales:
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
        if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
            self.yaw -= 1
        if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
            self.yaw += 1
        if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
            self.pitch += 1
            if self.pitch > 89:
                self.pitch = 89
        if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
            self.pitch -= 1
            if self.pitch < -89:
                self.pitch = -89

    def compute_frustum_corners_on_wall(self):
        half_width = self.image_plane_width / 2.0
        half_height = self.image_plane_height / 2.0
        corners = [
            np.array([-half_width, -half_height, self.image_plane_z]),
            np.array([ half_width, -half_height, self.image_plane_z]),
            np.array([ half_width,  half_height, self.image_plane_z]),
            np.array([-half_width,  half_height, self.image_plane_z])
        ]
        projected = []
        for corner in corners:
            ray_dir = corner - self.camera_position
            pt = ray_plane_intersection(self.camera_position, ray_dir, self.wall_plane_point, self.wall_plane_normal)
            if pt is not None:
                projected.append(pt)
        return projected

    def compute_detection_corners_on_wall(self):
        rect = self.det_rect
        corners = [
            np.array([rect["xmin"], rect["ymin"], self.image_plane_z]),
            np.array([rect["xmax"], rect["ymin"], self.image_plane_z]),
            np.array([rect["xmax"], rect["ymax"], self.image_plane_z]),
            np.array([rect["xmin"], rect["ymax"], self.image_plane_z])
        ]
        projected = []
        for corner in corners:
            ray_dir = corner - self.camera_position
            pt = ray_plane_intersection(self.camera_position, ray_dir, self.wall_plane_point, self.wall_plane_normal)
            if pt is not None:
                projected.append(pt)
        return projected

    def set_detection_rect(self, xmin, xmax, ymin, ymax):
        self.det_rect["xmin"] = xmin
        self.det_rect["xmax"] = xmax
        self.det_rect["ymin"] = ymin
        self.det_rect["ymax"] = ymax
        print("[OpenGL] Rectángulo de detección actualizado a:", self.det_rect)
    
    def render_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        import math
        # Definir el target de la cámara (por ejemplo, el centro de la escena en z=5)
        target = np.array([0, 0, 5])
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)
        dir_x = math.cos(rad_pitch) * math.sin(rad_yaw)
        dir_y = math.sin(rad_pitch)
        dir_z = math.cos(rad_pitch) * math.cos(rad_yaw)
        cam_pos = target - self.distance * np.array([dir_x, dir_y, dir_z])
        gluLookAt(cam_pos[0], cam_pos[1], cam_pos[2],
                  target[0], target[1], target[2],
                  0, 1, 0)
        
        # Dibujar ejes de referencia con la nueva orientación:
        glBegin(GL_LINES)
        # Eje rojo: ahora vertical, de (0,0,0) a (0,1,0)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)
        # Eje verde: en lugar de ir de (0,0,0) a (1,0,0), va de (0,0,0) a (-1,0,0)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(-1, 0, 0)
        # Eje azul se mantiene igual: de (0,0,0) a (0,0,1)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)
        glEnd()
        
        # Dibujar la pirámide de la cámara (rayos desde el origen a las esquinas proyectadas)
        frustum = self.compute_frustum_corners_on_wall()
        glColor3f(1, 1, 1)
        glBegin(GL_LINES)
        for corner in frustum:
            glVertex3fv(self.camera_position)
            glVertex3fv(corner)
        glEnd()
        glBegin(GL_LINE_LOOP)
        for corner in frustum:
            glVertex3fv(corner)
        glEnd()
        
        # Calcular el centroide del rectángulo de detección y dibujar un punto en él
        detection = self.compute_detection_corners_on_wall()
        if detection:
            centroid = np.mean(detection, axis=0)
            glColor3f(1, 0, 0)
            glPointSize(8)
            glBegin(GL_POINTS)
            glVertex3fv(centroid)
            glEnd()

    
    def main_loop(self):
        while not glfw.window_should_close(self.window):
            self.process_input()
            self.render_scene()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()



def convert_bbox_to_image_plane(x1, y1, x2, y2,
                                width_img=ANCHO_IMAGEN, height_img=LARGO_IMAGEN,
                                plane_width=2.0, plane_height=1.5):
    """
    Convierte un recuadro (en píxeles) al sistema de coordenadas del plano imagen.
    """
    x1_plane = (x1 / width_img - 0.5) * plane_width
    x2_plane = (x2 / width_img - 0.5) * plane_width
    y1_plane = (0.5 - y1 / height_img) * plane_height
    y2_plane = (0.5 - y2 / height_img) * plane_height
    xmin = min(x1_plane, x2_plane)
    xmax = max(x1_plane, x2_plane)
    ymin = min(y1_plane, y2_plane)
    ymax = max(y1_plane, y2_plane)
    return xmin, xmax, ymin, ymax
# ------------------- FIN CÓDIGO DE PROYECCIÓN ------------------- #

# --- Código existente del modelo y la cámara ---
class PinholeCamera:
    def __init__(self):
        self.position = np.zeros(4)
        self.k = np.zeros((3, 4))
        self.rt = np.zeros((4, 4))
        
def loadCamera():
    global myCamera
    myCamera = PinholeCamera()
    thetaY = 66 * DEGTORAD
    thetaZ = 0 * DEGTORAD
    thetaX = 0 * DEGTORAD
    R_y = np.array([(np.cos(thetaY), 0, -np.sin(thetaY)),
                    (0, 1, 0),
                    (np.sin(thetaY), 0, np.cos(thetaY))])
    R_z = np.array([(np.cos(thetaZ), -np.sin(thetaZ), 0),
                    (np.sin(thetaZ), np.cos(thetaZ), 0),
                    (0, 0, 1)])
    R_x = np.array([(1, 0, 0),
                    (0, np.cos(thetaX), np.sin(thetaX)),
                    (0, -np.sin(thetaX), np.cos(thetaX))])
    R_subt = np.dot(R_y, R_z)
    R_tot = np.dot(R_subt, R_x)
    T = np.array([(1, 0, 0, 0),
                  (0, 1, 0, 0),
                  (0, 0, 1, -252)])
    Res = np.dot(R_tot, T)
    RT = np.append(Res, [[0, 0, 0, 1]], axis=0)
    K = np.array([(FX, 0, CX, 0),
                  (0, FY, CY, 0),
                  (0, 0, 1, 0)])
    myCamera.rt = RT.copy()
    print(f"[DEBUG] Matriz RT de la cámara:\n{myCamera.rt}")
    myCamera.position = np.array([0, 0, -252, 1])
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
    p3d[:2] = p3d_h[:2] / p3d_h[2]
    if p3d_h[2] != 0:
        p3d = p3d_h / np.abs(p3d_h[2])
    else:
        p3d = p3d_h
    return p3d

def getIntersectionZ(p2d):
    p2d_h = np.array([p2d[0], p2d[1], 1])
    inv_K = np.linalg.inv(myCamera.k[:, :3])
    inv_RT = np.linalg.inv(myCamera.rt[:3, :3])
    p3d_h = np.dot(inv_K, p2d_h)
    p3d_h = np.dot(inv_RT, p3d_h)
    if np.abs(p3d_h[2]) > 1e-6:
        escala = -myCamera.position[2] / p3d_h[2]
        p3d_h *= escala
    else:
        p3d_h *= 1
    return np.array(p3d_h)

def calcular_distancia_3d(x_cam, y_cam, z_cam, x_punto, y_punto, z_punto):
    return np.sqrt((x_punto - x_cam)**2 + (y_punto - y_cam)**2 + (z_punto - z_cam)**2)
# --- FIN CÓDIGO DE CAMARA Y PROYECCIÓN ---

print("[INFO] Iniciando detección de fresas en el frame actual...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-max suppression")
    parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
    parser.add_argument("--webcam", type=int, default=1, help="1 = webcam, 0 = video")
    parser.add_argument("--directorio_video", type=str, help="Directorio al video")
    opt = parser.parse_args()
    
    # Configuración del dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device('cpu')))
    model.eval()
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Inicializar captura de video
    if opt.webcam == 1:
        cap = cv2.VideoCapture(2)
    else:
        cap = cv2.VideoCapture(opt.directorio_video)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara o el video.")
        exit(1)
    
    loadCamera()
    previous_positions = []
    threshold_distance = 5

    # Servidor XMLRPC
    server = SimpleXMLRPCServer(("0.0.0.0", 50000))
    server.RequestHandlerClass.protocol_version = "HTTP/1.1"
    print("Servidor XML-RPC corriendo en el puerto 50000...")

    def get_next_pose():
        if detected_points:
            pose = detected_points[-1]
            print(f"[DEBUG] Enviando última posición detectada: {pose}")
            return [pose[0], pose[1], pose[2], 0, 0, 0]
        else:
            print("[DEBUG] No se detectaron puntos, enviando posición de inicio")
            return [0, 0, 0, 0, 0, 0]
    server.register_function(get_next_pose, "get_next_pose")
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    # --- Iniciar el procesamiento de detección en un hilo separado ---
    def detection_loop():
        global detected_points, previous_positions
        while True:
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
            for detection in detections:
                if detection is not None:
                    detection = rescale_boxes(detection, opt.img_size, frame.shape[:2])
                    # Actualizar el recuadro de detección en OpenGL con la primera caja detectada
                    if len(detection) > 0:
                        x1, y1, x2, y2, cls_conf, conf, cls_pred = detection[0]
                        xmin_plane, xmax_plane, ymin_plane, ymax_plane = convert_bbox_to_image_plane(x1, y1, x2, y2)
                        cp.set_detection_rect(xmin_plane, xmax_plane, ymin_plane, ymax_plane)
                    for i, det in enumerate(detection):
                        if len(det) == 7:
                            x1, y1, x2, y2, cls_conf, conf, cls_pred = det
                            box_w = x2 - x1
                            box_h = y2 - y1
                            center_x = x1 + box_w // 2
                            center_y = y1 + box_h // 2
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            label = f"Fresa - P{i+1}"
                            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            p2d = np.array([center_x, center_y])
                            pixelOnGround3D = getIntersectionZ(p2d)
                            x_punto = float(pixelOnGround3D[0])
                            y_punto = float(pixelOnGround3D[1])
                            z_punto = float(pixelOnGround3D[2])
                            positions.append((x_punto, y_punto, z_punto))
            # Filtrar posiciones similares
            for pos in positions:
                is_new = True
                for prev in previous_positions:
                    if calcular_distancia_3d(prev[0], prev[1], prev[2], pos[0], pos[1], pos[2]) < threshold_distance:
                        is_new = False
                        break
                if is_new:
                    previous_positions.append(pos)
                    detected_points.append(pos)
                    print(f"Nueva detección: {pos}")
            # Opcional: mostrar la ventana de detección
            cv2.imshow('Detección de Fresas', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    detection_thread = threading.Thread(target=detection_loop)
    detection_thread.daemon = True
    detection_thread.start()
    # --- FIN del procesamiento de detección ---

    # Instanciar la proyección y ejecutar el loop de OpenGL en el hilo principal
    cp = CameraProjection(width=800, height=600)
    cp.main_loop()

