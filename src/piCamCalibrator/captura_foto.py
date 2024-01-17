import cv2
import os

def main():
    # Establecer la carpeta en la que se guardan las imágenes
    folder_name = "/home/dcampoamor/Escritorio/chess_board"

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    # Contador para enumerar las fotos
    count = 1

    while count <= 20:
        # Capturar un fotograma de la cámara
        ret, frame = cap.read()

        # Mostrar el fotograma en una ventana (opcional)
        cv2.imshow("Chess Board Capture", frame)

        # Esperar a que se presione la tecla 'f' para capturar la foto
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            # Guardar la foto en la carpeta
            file_name = os.path.join(folder_name, f"{count}.png")
            cv2.imwrite(file_name, frame)
            print(f"Foto {count} guardada en {file_name}")
            count += 1

    # Liberar la cámara y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()

    print("Se han tomado y guardado 20 fotos en la carpeta 'chess_board'.")
    
if __name__ == "__main__":
    main()

