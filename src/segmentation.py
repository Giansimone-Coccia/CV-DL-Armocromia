import os
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

            vis_img_pil.save(os.path.join(directory,f'result_{filename.split("_")[1]}_{filename.split("_")[2]}'))
