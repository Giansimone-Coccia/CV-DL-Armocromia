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
        # Inizializza una variabile per i volti
        faces = None
        
        # Itera attraverso i file nella directory 'results/faces'
        for filename in os.listdir(os.path.join(self._project_dir, 'results/faces')):
            # Legge l'immagine corrente e la converte in un tensore per elaborazione
            image = facer.hwc2bchw(facer.read_hwc(f'results/faces/{filename}')).to(device=self._device)
            
            # Esegue il rilevamento dei volti sull'immagine
            with torch.inference_mode():
                faces = self._face_detector(image)
            facer.show_bchw(facer.draw_bchw(image, faces))

            # Esegue l'analisi dei volti sull'immagine
            with torch.inference_mode():
                faces = self._face_parser(image, faces)

            # Ottiene le previsioni di segmentazione dai volti
            seg_logits = faces['seg']['logits']
            seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
            
            # Calcola il numero di classi di segmentazione
            n_classes = seg_probs.size(1)

            all_segments = []
            for face_id in range(seg_probs.size(0)):
                for class_id in range(n_classes):
                    mask = (seg_probs[face_id, class_id] > 0.5).float()
                    if mask.sum() > 0:
                        mask_pil = to_pil_image(mask.cpu())
                        all_segments.append((face_id, class_id, mask))

            '''for (face_id, class_id, mask_pil) in all_segments:
                mask_pil.show(title=f'Face {face_id} - Class {class_id}')'''
            
            # Calcola un'immagine visibile basata sulle previsioni di segmentazione
            vis_seg_probs = seg_probs.argmax(dim=1).float() / n_classes * 255
            vis_img = vis_seg_probs.sum(0, keepdim=True)
            facer.show_bhw(vis_img)
            facer.show_bchw(facer.draw_bchw(image, faces))

            # Converte il tensore in un'immagine PIL
            vis_img_pil = to_pil_image(vis_img.byte())

            # Crea una directory per le immagini risultanti se non esiste già
            directory = "results/faces_facer"
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Salva l'immagine risultante nella directory 'results/faces_facer', rinominandola
            vis_img_pil.save(os.path.join(directory,f'result_{filename.split("_")[1]}_{filename.split("_")[2]}'), format='JPEG')

        # Restituisce l'oggetto faces
        return all_segments
    
    def extract_dominant_colors(self, all_segments, filename='result_0_0.jpg'):
        dominant_colors = {}
        image_path = os.path.join(self._project_dir, 'results/faces', filename)
        image = facer.read_hwc(image_path)
        image_tensor = facer.hwc2bchw(image).to(device=self._device)

        for part_segmented in all_segments:
            # Get the coordinates where the segmentation mask is non-zero
            pixel_coords = torch.nonzero(part_segmented[2])
            segmented_colors = []
            for coord in pixel_coords:
                color = image_tensor[0, :, coord[0], coord[1]].cpu().numpy()
                segmented_colors.append(color)
            if len(segmented_colors) > 0:
                segmented_colors = np.array(segmented_colors)
                kmeans = KMeans(n_clusters=3).fit(segmented_colors)
                dominant_colors[part_segmented[1]] = kmeans.cluster_centers_

        return dominant_colors