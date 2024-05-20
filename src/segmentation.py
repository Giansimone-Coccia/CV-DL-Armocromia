import os
from tkinter import Image
import numpy as np
from sklearn.cluster import KMeans
import torch
from torchvision.transforms.functional import to_pil_image
import facer

class Segmentation:
    def __init__(self):
        self._project_dir = os.getcwd()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._face_detector = facer.face_detector('retinaface/mobilenet', device=self._device)
        self._face_parser = facer.face_parser('farl/lapa/448', device=self._device)

    def process_images(self):
        faces = None
        for filename in os.listdir(os.path.join(self._project_dir, 'results/faces')):
            image = facer.hwc2bchw(facer.read_hwc(f'results/faces/{filename}')).to(device=self._device)
            with torch.inference_mode():
                faces = self._face_detector(image)
            
            with torch.inference_mode():
                faces = self._face_parser(image, faces)

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

            vis_img_pil.save(os.path.join(directory,f'result_{filename.split("_")[1]}_{filename.split("_")[2]}.jpg'))
        return faces
        
    def extract_dominant_colors(self, faces, filename='result_0_0.jpg'):
        dominant_colors = {}

        # Carica l'immagine
        image_path = os.path.join(self._project_dir, 'results/faces', filename)
        image_tensor = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=self._device)

        seg_logits = faces['seg']['logits']
        for i, part_segmented in enumerate(seg_logits):  # Itera sui tensori di segmentazione per ogni parte del viso
            seg_probs_part = seg_logits[i].softmax(dim=0)  # Applica softmax lungo la dimensione delle classi
            seg_mask_part = seg_probs_part.argmax(dim=0)  # Ottieni la maschera segmentata per questa parte del viso
            pixel_coords = torch.nonzero(seg_mask_part)  # Trova le coordinate dei pixel segmentati

            segmented_colors = []
            for coord in pixel_coords:
                color = image_tensor[0, :, coord[0], coord[1]].cpu().numpy()  # Ottieni il colore del pixel dall'immagine originale
                segmented_colors.append(color)

            segmented_colors = np.array(segmented_colors)

            # Usa KMeans con 3 cluster per trovare i 3 colori dominanti
            kmeans = KMeans(n_clusters=3).fit(segmented_colors)
            dominant_colors[i] = kmeans.cluster_centers_

        return dominant_colors