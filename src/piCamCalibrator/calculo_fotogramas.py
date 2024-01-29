import cv2

def calculate_fps(capture):
    # Obtiene la resolución de la cámara
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (width, height)
    
    # Inicializa el reloj
    start_time = cv2.getTickCount()
    frame_count = 0

    # Lee cada fotograma y cuenta los cuadros por segundo
    while True:
        _, frame = capture.read()
        if frame is None:
            break

        frame_count += 1

        # Muestra el fotograma
        cv2.imshow("Frame", frame)

        # Espera 1 milisegundo y verifica si se presionó la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calcula el tiempo transcurrido
    end_time = cv2.getTickCount()
    elapsed_time = (end_time - start_time) / cv2.getTickFrequency()

    # Calcula los FPS
    fps = frame_count / elapsed_time

    # Calcula los megapíxeles
    megapixels = calculate_megapixels(resolution)

    return fps, resolution, megapixels

def calculate_megapixels(resolution):
    # Calcula los megapíxeles (1 megapíxel = 1,000,000 píxeles)
    megapixels = (resolution[0] * resolution[1]) / 1e6
    return megapixels

def main():
    # Pregunta al usuario qué cámara desea usar
    camera_choice = input("¿Qué cámara deseas utilizar? (1 para webcam integrada, 2 para cámara USB): ")

    # Inicializa la variable de captura
    capture = None

    if camera_choice == '1':
        # Cierra todas las ventanas abiertas
        cv2.destroyAllWindows()

        # Usa la webcam integrada
        capture = cv2.VideoCapture(2)
    elif camera_choice == '2':
        # Cierra todas las ventanas abiertas
        cv2.destroyAllWindows()

        # Usa la cámara USB (puedes ajustar el índice según sea necesario)
        usb_camera_index = 1
        capture = cv2.VideoCapture(usb_camera_index)
    else:
        print("Opción no válida. Saliendo del programa.")
        return

    if not capture.isOpened():
        print("No se puede acceder a la cámara seleccionada. Saliendo del programa.")
        return

    # Calcula los FPS, la resolución y los megapíxeles y muestra el contenido de la cámara
    fps, resolution, megapixels = calculate_fps(capture)

    # Muestra los resultados
    print(f"Los FPS de la cámara seleccionada son: {fps:.2f}")
    print(f"Resolución utilizada: {resolution[0]}x{resolution[1]}")
    print(f"Número de megapíxeles: {megapixels:.2f} MP")

    # Libera la captura y cierra la ventana de la cámara
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


