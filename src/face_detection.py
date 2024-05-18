import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO

class FaceDetection:
    def __init__(self, model_path='data/models/yolov8l-face.pt'):
        self.model_path = model_path
        assert os.path.exists(model_path), "Il percorso del modello non è valido"
        self.model = YOLO(model_path)

    def face_detection(self, image_path, output_dir='results/faces'):
        # Esegui l'inferenza sull'immagine con YOLOv8
        results = self.model(image_path)

        # Itera attraverso tutte le bounding box individuate
        for i, result in enumerate(results):
            boxes = result.boxes.data  # Boxes object for bounding box outputs

            result.save(filename='results/result.jpg')
            img = mpimg.imread('results/result.jpg')
            plt.imshow(img)

            for j, box in enumerate(boxes):
                # Ottieni le coordinate della bounding box
                x_min, y_min, x_max, y_max, conf, cls = box.tolist()[:6]

                # Ritaglia l'area corrispondente dall'immagine originale
                img = Image.open(image_path)
                cropped_img = img.crop((x_min, y_min, x_max, y_max))

                # Salva l'immagine ritagliata
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                cropped_img.save(os.path.join(output_dir, f'result_{i}_{"0" * (len(str(len(boxes))) - len(str(j)))}{j}.jpg'))
