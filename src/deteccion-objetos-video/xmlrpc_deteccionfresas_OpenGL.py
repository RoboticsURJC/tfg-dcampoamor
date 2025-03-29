#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import xmlrpc.client
from xmlrpc.server import SimpleXMLRPCServer
import torch
import argparse
import threading
import glfw
from OpenGL.GL import *        # Importa funciones y constantes de OpenGL
from OpenGL.GLU import *
from models import Darknet
from utils.utils import load_classes, rescale_boxes, non_max_suppression
from utils.datasets import pad_to_square, resize
import torchvision.transforms as transforms
from torch.autograd import Variable
import math

# -------------------------------
# Parámetros de la imagen y de la cámara (para calibración)
ANCHO_IMAGEN = 640
LARGO_IMAGEN = 480
FX = 816.218
FY = 814.443
CX = 316.068
CY = 236.933
DEGTORAD = math.pi / 180

detected_points = []  # Para enviar posiciones vía XMLRPC
stop_server = False

print("[INFO] Iniciando programa...")

# =============================================================================
# Funciones para la proyección 3D usando el modelo pinhole
# =============================================================================
def ray_plane_intersection(ray_origin, ray_direction, plane_point, plane_normal):
    denom = np.dot(ray_direction, plane_normal)
    if abs(denom) < 1e-6:
        return None
    t = np.dot(plane_point - ray_origin, plane_normal) / denom
    if t < 0:
        return None
    return ray_origin + t * ray_direction

def pixel2optical(p2d):
    # Ajusta las coordenadas según el sistema del programa original
    aux = p2d[0]
    p2d[0] = LARGO_IMAGEN - 1 - p2d[1]
    p2d[1] = aux
    return p2d

def backproject_pixel_to_3d(px, py):
    """
    Dado un píxel (px,py), calcula la intersección del rayo correspondiente con el plano de la mesa (z=0)
    usando el modelo pinhole y los parámetros calibrados en myCamera.
    """
    p2d = pixel2optical([px, py])
    p2d_h = np.array([p2d[0], p2d[1], 1.0])
    inv_K = np.linalg.inv(myCamera.k[:, :3])
    inv_R = np.linalg.inv(myCamera.rt[:3, :3])
    ray_dir = np.dot(inv_K, p2d_h)
    ray_dir = np.dot(inv_R, ray_dir)
    cam_pos = myCamera.position[:3]  # La cámara está en [0, 0, -252]
    if abs(ray_dir[2]) < 1e-6:
        return None
    factor = -cam_pos[2] / ray_dir[2]  # Para que la intersección tenga z=0
    intersection = cam_pos + factor * ray_dir
    return intersection

def get_3d_polygon(x1, y1, x2, y2):
    """
    Retorna una lista de 4 puntos 3D (en el plano de la mesa, z=0) obtenidos proyectando las 4 esquinas
    del bounding box.
    """
    corners_pix = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    poly = []
    for (px, py) in corners_pix:
        p3d = backproject_pixel_to_3d(px, py)
        if p3d is not None:
            poly.append(p3d)
    return poly

def calcular_distancia_3d(x_cam, y_cam, z_cam, x_p, y_p, z_p):
    return math.sqrt((x_p - x_cam)**2 + (y_p - y_cam)**2 + (z_p - z_cam)**2)

# =============================================================================
# Funciones y clase para la cámara pinhole (modelo original)
# =============================================================================
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
    R_y = np.array([
        (math.cos(thetaY), 0, -math.sin(thetaY)),
        (0, 1, 0),
        (math.sin(thetaY), 0, math.cos(thetaY))
    ])
    R_z = np.array([
        (math.cos(thetaZ), -math.sin(thetaZ), 0),
        (math.sin(thetaZ),  math.cos(thetaZ), 0),
        (0, 0, 1)
    ])
    R_x = np.array([
        (1, 0, 0),
        (0, math.cos(thetaX), math.sin(thetaX)),
        (0, -math.sin(thetaX), math.cos(thetaX))
    ])
    R_subt = np.dot(R_y, R_z)
    R_tot = np.dot(R_subt, R_x)
    # La cámara está a 252 mm sobre la mesa, por lo tanto T desplaza en z: -252 (mesa en z=0)
    T = np.array([
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, -252)
    ])
    Res = np.dot(R_tot, T)
    RT = np.append(Res, [[0, 0, 0, 1]], axis=0)
    K = np.array([
        (FX, 0, CX, 0),
        (0, FY, CY, 0),
        (0, 0, 1, 0)
    ])
    myCamera.rt = RT.copy()
    myCamera.position = np.array([0, 0, -252, 1])
    myCamera.k = K
    myCamera.rt = RT
    print("[DEBUG] Matriz RT de la cámara:\n", myCamera.rt)
    print("[INFO] Cámara cargada. Altura real: 252 mm sobre la mesa.")

# =============================================================================
# Clase OpenGL para visualizar la escena y permitir navegación interactiva
# =============================================================================
class PyramidProjection:
    def __init__(self, width=800, height=600):
        if not glfw.init():
            raise Exception("No se pudo inicializar GLFW.")
        self.width = width
        self.height = height
        self.window = glfw.create_window(width, height, "Detección y Proyección 3D", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("No se pudo crear la ventana GLFW.")
        glfw.make_context_current(self.window)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        self.init_gl()

        # Para traslación lateral (pan)
        self.pan = np.array([0.0, 0.0, 0.0])

        # La posición de la cámara en el mundo se toma de myCamera
        self.camera_position = myCamera.position[:3]  # [0,0,-252]
        # El plano imagen (lo que ve la cámara) se define en z = 1.0 (en el sistema de la cámara)
        self.image_plane_z = 1.0
        self.image_plane_width = 2.0
        self.image_plane_height = 1.5
        half_w = self.image_plane_width / 2.0
        half_h = self.image_plane_height / 2.0
        self.base_corners = [
            np.array([-half_w, -half_h, self.image_plane_z]),
            np.array([ half_w, -half_h, self.image_plane_z]),
            np.array([ half_w,  half_h, self.image_plane_z]),
            np.array([-half_w,  half_h, self.image_plane_z])
        ]
        # Calcular el polígono completo de la vista (proyección de las 4 esquinas del sensor)
        self.full_view_polygon = self.compute_full_view_polygon()

        # Variables para la cámara virtual de OpenGL
        self.yaw = 0
        self.pitch = 0
        self.distance = 800.0  # En mm
        self.is_dragging = False
        self.last_cursor_x = 0
        self.last_cursor_y = 0

        # Textura para la imagen de la cámara
        self.camera_texture = glGenTextures(1)
        self.latest_frame = None

        # Polígono de detección (proyección 3D irregular del bounding box) – se dibuja en azul
        self.detection_polygon_3d = None

    def init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, self.width/self.height, 1.0, 3000.0)
        glMatrixMode(GL_MODELVIEW)

    def scroll_callback(self, window, xoffset, yoffset):
        self.distance -= yoffset * 10.0
        if self.distance < 100:
            self.distance = 100

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
            self.yaw += dx * 0.5
            self.pitch -= dy * 0.5
            if self.pitch > 89: self.pitch = 89
            if self.pitch < -89: self.pitch = -89
            self.last_cursor_x = xpos
            self.last_cursor_y = ypos

    def process_input(self):
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
        if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
            self.yaw -= 1
        if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
            self.yaw += 1
        if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
            self.pitch += 1
            if self.pitch > 89: self.pitch = 89
        if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
            self.pitch -= 1
            if self.pitch < -89: self.pitch = -89
        # Teclas para traslación lateral
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.pan[0] -= 10.0
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.pan[0] += 10.0

    def update_camera_texture(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.flip(rgb, 0)
        glBindTexture(GL_TEXTURE_2D, self.camera_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        h, w, ch = rgb.shape
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb)
        glBindTexture(GL_TEXTURE_2D, 0)

    def compute_full_view_polygon(self):
        """
        Proyecta las 4 esquinas de la imagen completa (píxeles: (0,0), (ANCHO,0),
        (ANCHO,LARGO), (0,LARGO)) sobre el plano de la mesa (z=0) usando el modelo pinhole.
        """
        corners_pix = [(0,0), (ANCHO_IMAGEN, 0), (ANCHO_IMAGEN, LARGO_IMAGEN), (0, LARGO_IMAGEN)]
        poly = []
        for (px, py) in corners_pix:
            p3d = backproject_pixel_to_3d(px, py)
            if p3d is not None:
                poly.append(p3d)
        return poly

    def set_detection_polygon(self, corners_3d):
        if len(corners_3d) == 4:
            self.detection_polygon_3d = corners_3d
        else:
            self.detection_polygon_3d = None

    def draw_camera_axes(self):
        """
        Dibuja los ejes de la cámara en el mundo usando la matriz de rotación de myCamera.
        Se asume que: eje X en rojo, eje Y en verde, eje Z en azul (donde X x Y = Z).
        """
        R = myCamera.rt[:3, :3]
        # Suponemos que las filas de R representan los ejes en el sistema de la cámara
        x_axis = R[0, :]
        y_axis = R[1, :]
        z_axis = R[2, :]
        origin = self.camera_position
        axis_length = 50.0  # Longitud de los ejes en mm
        glLineWidth(3)
        glBegin(GL_LINES)
        glColor3f(1,0,0)  # Eje X en rojo
        glVertex3fv(origin)
        glVertex3fv(origin + axis_length * x_axis)
        glColor3f(0,1,0)  # Eje Y en verde
        glVertex3fv(origin)
        glVertex3fv(origin + axis_length * y_axis)
        glColor3f(0,0,1)  # Eje Z en azul
        glVertex3fv(origin)
        glVertex3fv(origin + axis_length * z_axis)
        glEnd()

    def render_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Centro de la escena se traslada según pan
        center = self.pan.copy()
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)
        dir_vec = np.array([
            math.cos(rad_pitch)*math.sin(rad_yaw),
            math.sin(rad_pitch),
            math.cos(rad_pitch)*math.cos(rad_yaw)
        ])
        eye = center - self.distance * dir_vec
        gluLookAt(eye[0], eye[1], eye[2],
                  center[0], center[1], center[2],
                  0, 0, 1)

        # Dibujar el contorno completo del plano de la cámara (proyección de la imagen completa)
        full_poly = self.compute_full_view_polygon()
        if full_poly is not None and len(full_poly) == 4:
            glColor3f(1,1,1)  # blanco
            glLineWidth(2)
            glBegin(GL_LINE_LOOP)
            for pt in full_poly:
                glVertex3fv(pt)
            glEnd()
            # Unir cada esquina con la posición de la cámara (frustum)
            glBegin(GL_LINES)
            for pt in full_poly:
                glVertex3fv(self.camera_position)
                glVertex3fv(pt)
            glEnd()

        # Dibujar la base de la pirámide texturizada (lo que ve la cámara)
        glEnable(GL_TEXTURE_2D)
        if self.latest_frame is not None:
            self.update_camera_texture(self.latest_frame)
        glBindTexture(GL_TEXTURE_2D, self.camera_texture)
        glColor3f(1,1,1)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3fv(self.base_corners[0])
        glTexCoord2f(1.0, 0.0); glVertex3fv(self.base_corners[1])
        glTexCoord2f(1.0, 1.0); glVertex3fv(self.base_corners[2])
        glTexCoord2f(0.0, 1.0); glVertex3fv(self.base_corners[3])
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        self.latest_frame = None

        # Dibujar el polígono de detección proyectado (en azul)
        if self.detection_polygon_3d is not None:
            glColor3f(0,0,1)  # azul
            glLineWidth(3)
            glBegin(GL_LINE_LOOP)
            for pt in self.detection_polygon_3d:
                glVertex3fv(pt)
            glEnd()

        # Dibujar los ejes de la cámara
        self.draw_camera_axes()

    def main_loop(self):
        while not glfw.window_should_close(self.window):
            self.process_input()
            self.render_scene()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()

    def process_input(self):
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
        if glfw.get_key(self.window, glfw.KEY_LEFT) == glfw.PRESS:
            self.yaw -= 1
        if glfw.get_key(self.window, glfw.KEY_RIGHT) == glfw.PRESS:
            self.yaw += 1
        if glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS:
            self.pitch += 1
            if self.pitch > 89: self.pitch = 89
        if glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS:
            self.pitch -= 1
            if self.pitch < -89: self.pitch = -89
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.pan[0] -= 10.0
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.pan[0] += 10.0

# =============================================================================
# Programa principal
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="Ruta al modelo YOLO")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_99.pth", help="Ruta a los pesos")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="Ruta a las clases")
    parser.add_argument("--conf_thres", type=float, default=0.85, help="Umbral de confianza YOLO")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="Umbral NMS")
    parser.add_argument("--img_size", type=int, default=320, help="Tamaño de la imagen YOLO")
    parser.add_argument("--webcam", type=int, default=1, help="1 = usar webcam, 0 = usar video")
    parser.add_argument("--directorio_video", type=str, help="Ruta al video (si no se usa webcam)")
    parser.add_argument("--camera_index", type=int, default=2, help="Índice de la cámara USB (default=1)")
    opt = parser.parse_args()

    # Configurar dispositivo y cargar modelo YOLO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path, map_location=device))
    model.eval()
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Iniciar captura de video (cámara USB o video)
    if opt.webcam == 1:
        print(f"[INFO] Abriendo cámara USB en índice {opt.camera_index}")
        cap = cv2.VideoCapture(opt.camera_index)
    else:
        if not opt.directorio_video:
            print("[ERROR] No se especificó --directorio_video")
            exit(1)
        print(f"[INFO] Abriendo video: {opt.directorio_video}")
        cap = cv2.VideoCapture(opt.directorio_video)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara o el video.")
        exit(1)

    loadCamera()
    print("[INFO] Cámara cargada. Altura real: 252 mm sobre la mesa.")

    # Servidor XMLRPC (opcional)
    server = SimpleXMLRPCServer(("0.0.0.0", 50000))
    server.RequestHandlerClass.protocol_version = "HTTP/1.1"
    print("[INFO] Servidor XML-RPC corriendo en el puerto 50000...")
    def get_next_pose():
        if detected_points:
            pose = detected_points[-1]
            print("[DEBUG] Última detección:", pose)
            return [pose[0], pose[1], pose[2], 0,0,0]
        else:
            print("[DEBUG] Sin detecciones, enviando 0,0,0")
            return [0,0,0,0,0,0]
    server.register_function(get_next_pose, "get_next_pose")
    xmlrpc_thr = threading.Thread(target=server.serve_forever)
    xmlrpc_thr.daemon = True
    xmlrpc_thr.start()

    # Instanciar la ventana OpenGL
    pp = PyramidProjection(width=800, height=600)

    def detection_loop():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] No se pudo leer frame de la cámara/video.")
                break
            # Actualizar el frame para la textura en OpenGL
            pp.latest_frame = frame.copy()

            # Procesar detección YOLO
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgTensor = transforms.ToTensor()(rgb)
            imgTensor, _ = pad_to_square(imgTensor, 0)
            imgTensor = resize(imgTensor, opt.img_size)
            imgTensor = imgTensor.unsqueeze(0)
            imgTensor = imgTensor.type(Tensor)
            with torch.no_grad():
                outs = model(imgTensor)
                outs = non_max_suppression(outs, opt.conf_thres, opt.nms_thres)
            for det in outs:
                if det is not None and len(det) > 0:
                    det = rescale_boxes(det, opt.img_size, frame.shape[:2])
                    x1, y1, x2, y2, obj_conf, cls_conf, cls_pred = det[0]
                    real_conf = float(obj_conf)  # O usar obj_conf * cls_conf
                    class_name = classes[int(cls_pred)]
                    poly = get_3d_polygon(x1, y1, x2, y2)
                    if len(poly) == 4:
                        pp.set_detection_polygon(poly)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    center3d = backproject_pixel_to_3d(cx, cy)
                    dist = 999.9
                    if center3d is not None:
                        dist = calcular_distancia_3d(myCamera.position[0], myCamera.position[1], myCamera.position[2],
                                                     center3d[0], center3d[1], center3d[2])
                    print(f"Detección: {class_name}, conf={real_conf:.2f}, centro=({cx:.1f},{cy:.1f}), dist={dist:.1f} mm")
                    # Dibujar bounding box en OpenCV en azul
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                    cv2.putText(frame, f"{class_name} {real_conf:.2f}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.imshow("Deteccion de Fresas", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    detection_thr = threading.Thread(target=detection_loop)
    detection_thr.daemon = True
    detection_thr.start()

    pp.main_loop()

