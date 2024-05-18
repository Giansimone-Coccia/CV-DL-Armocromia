import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO

class FaceDetection:
    def __init__(self):
        # Ottieni il percorso completo del file
        self._project_dir = os.getcwd()
        # Percorso del modello
        self._model_path = os.path.join(self._project_dir, 'data/models/yolov8l-face.pt')
        # Assicurati che il percorso del modello sia corretto
        assert os.path.exists(self._model_path), f"Il percorso del modello non è valido"
        # Run inference on an image with YOLOv8
        self._model = YOLO(self._model_path)
        self._results = self._model(os.path.join(self._project_dir, 'data/images/Faces.jpg'))

    def face_detection(self, output_dir='results/faces'):
        # Esegui l'inferenza sull'immagine con YOLOv8
        #results = self._model(image_path)

        # Itera attraverso tutte le bounding box individuate
        for i, result in enumerate(self._results):
            boxes = result.boxes.data  # Boxes object for bounding box outputs

            result.save(filename='results/result.jpg')
            img = mpimg.imread('results/result.jpg')
            plt.imshow(img)

            for j, box in enumerate(boxes):
                # Ottieni le coordinate della bounding box
                x_min, y_min, x_max, y_max, conf, cls = box.tolist()[:6]

                # Ritaglia l'area corrispondente dall'immagine originale
                img = Image.open(os.path.join(self._project_dir, 'data/images/Faces.jpg'))
                cropped_img = img.crop((x_min, y_min, x_max, y_max))

                # Salva l'immagine ritagliata
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                cropped_img.save(os.path.join(output_dir, f'result_{i}_{"0" * (len(str(len(boxes))) - len(str(j)))}{j}.jpg'))
