import cv2, time, os, tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(30)

class Detector:
    def __init__(self):
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList =f.read().splitlines()

        #Colors list
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def downloadModel(self, modelURL):

        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]

        self.cacheDir = "./pretrained_models"

        os.makedirs(self.cacheDir, exist_ok=True)
        
        get_file(fname=fileName,
        origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        print("Loading Model " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir,"checkpoints",self.modelName,"saved_model"))

        print("Model " + self.modelName + " loaded successfuly...")

    def createBoundingBox(self, image, threshold = 0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]

        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        #Definimos el tamaño de la altura, la anchura y el número de canales de la imagen.
        #El canal se refiere las capas individuales que componen una imagen 
        #(En una imagen en color hay tres canales principales: rojo, verde y azul, que se utilizan para crear la imagen completa)
        imH,imW,imC = image.shape

        #Limitamos el número de bounding boxes a un máximo de 10 por imagen, Umbral IOU al 50 % y el Umbral de Puntuación al 50% también.
        #IOU Threshold (Umbral de Intersección sobre Unión): umbral de similitud que se utiliza para evaluar la precisión de un modelo de detección de objetos en la detección y localización de objetos en una imagen.
        #Score Threshold: valor mínimo de confianza que se requiere para considerar que un objeto ha sido detectado en una imagen.
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size = 10, iou_threshold=threshold, score_threshold=0.5)

        print(bboxIdx)
       
        if len(bboxIdx) !=0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.classesList[classIndex].upper()
                classColor = self.colorList[classIndex]

                displayText = '{}: {}%'.format(classLabelText, classConfidence)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH )
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                print(ymin, xmin, ymax, xmax)
                
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1 )
                cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                ################################################################################
                lineWidth = min(int((xmax - xmin)* 0.2) , int((ymax - ymin)* 0.2))

                cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
                cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)

                cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
                cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)
                ################################################################################
               
                cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=5)

                cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)

        return image
    
    
    def predictImage(self, imagePath, threshold = 0.5):
        image = cv2.imread(imagePath)

        bboxImage = self.createBoundingBox(image)
        
        self.createBoundingBox(image)

        cv2.imwrite(self.modelName + ".jpeg", bboxImage)
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyallWindows()
    
    def predictVideo(self, videoPath, threshold = 0.5):
        cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened() == False):
            print("Error opening file...")
            return
        
        (success, image) = cap.read()
        startTime = 0

        while success:
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime

            bboxImage = self.createBoundingBox(image, threshold)

            cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            cv2.imshow("Result", bboxImage)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            (success, image) = cap.read()
        cv2.destroyAllWindows()