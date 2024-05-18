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
        # Inizializza il modello YOLO
        self._model = YOLO(self._model_path)
        print("Inizializzazione...")

    def face_detection(self, input_dir='data/images', output_dir='results/faces'):
        # Elenco dei file nella cartella di input
        image_files = os.listdir(input_dir)

        # Assicurati che l'output_dir esista, altrimenti crealo
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("Directory creata")

        # Itera su ogni file di immagine nella cartella di input
        for image_file in image_files:
            print(f"Immagine: {image_file}")
            # Percorso completo dell'immagine di input
            image_path = os.path.join(input_dir, image_file)
            
            # Esegui l'inferenza sull'immagine con YOLOv8
            results = self._model(image_path)

            # Itera attraverso tutte le bounding box individuate
            for i, result in enumerate(results):
                boxes = result.boxes.data  # Boxes object for bounding box outputs

                """ result.save(filename='results/result.jpg')
                img = mpimg.imread('results/result.jpg')
                plt.imshow(img) """

                for j, box in enumerate(boxes):
                    # Ottieni le coordinate della bounding box
                    x_min, y_min, x_max, y_max, conf, cls = box.tolist()[:6]

                    # Ritaglia l'area corrispondente dall'immagine originale
                    img = Image.open(image_path)
                    cropped_img = img.crop((x_min, y_min, x_max, y_max))

                    # Salva l'immagine ritagliata
                    output_file = f'result_{i}_{j}.jpg'
                    cropped_img.save(os.path.join(output_dir, output_file))
        print("Processo concluso con successo")
