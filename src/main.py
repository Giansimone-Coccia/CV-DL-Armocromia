import os
import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch 
import facer #pip install pyfacer      pip install timm
from torchvision.transforms.functional import to_pil_image
import shutil

# Ottieni il percorso completo del file
project_dir = os.getcwd()

# Percorso del modello
model_path = os.path.join(project_dir, 'data/models/yolov8l-face.pt')

# Assicurati che il percorso del modello sia corretto
assert os.path.exists(model_path), f"Il percorso del modello non è valido"
# Run inference on an image with YOLOv8
model = YOLO(model_path)
results = model(os.path.join(project_dir, 'data/images/Faces.jpg'))

for i, result in enumerate(results):
    boxes = result.boxes.data  # Boxes object for bounding box outputs

    result.save(filename='results/result.jpg')
    img = mpimg.imread('results/result.jpg')
    plt.imshow(img)
    # Itera attraverso tutte le bounding box individuate
    for j, box in enumerate(boxes):
        # Ottieni le coordinate della bounding box
        x_min, y_min, x_max, y_max, conf, cls = box.tolist()[:6]

        # Ritaglia l'area corrispondente dall'immagine originale
        img = Image.open(os.path.join(project_dir, 'data/images/Faces.jpg'))
        cropped_img = img.crop((x_min, y_min, x_max, y_max))

        # Salva l'immagine ritagliata
        directory = "results/faces"
        # Controllo se la cartella esiste, altrimenti la creo
        if not os.path.exists(directory):
            os.makedirs(directory)
        #shutil.rmtree(directory)
        cropped_img.save(os.path.join(directory,f'result_{i}_{"0" * (len(str(len(boxes))) - len(str(j)))}{j}.jpg'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_parser = facer.face_parser('farl/lapa/448', device=device)

for filename in os.listdir(os.path.join(project_dir, 'results/faces')):
    image = facer.hwc2bchw(facer.read_hwc(f'results/faces/{filename}')).to(device=device)
    with torch.inference_mode():
        faces = face_detector(image)
    
    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    n_classes = seg_probs.size(1)
    vis_seg_probs = seg_probs.argmax(dim=1).float()/n_classes*255
    vis_img = vis_seg_probs.sum(0, keepdim=True)

     # Converti il tensor in un'immagine PIL
    vis_img_pil = to_pil_image(vis_img.byte())

    directory = "results/faces_facer"

    if not os.path.exists(directory):
            os.makedirs(directory)
            
    vis_img_pil.save(os.path.join(directory,f'result_{filename.split("_")[1]}_{filename.split("_")[2]}'))