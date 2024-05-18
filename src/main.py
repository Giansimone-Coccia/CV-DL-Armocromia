from face_detection import FaceDetection
from segmentation import Segmentation


fd = FaceDetection()
fd.face_detection('C:\\Users\\gians_ji5genm\\OneDrive - Università Politecnica delle Marche\\UNIVPM\\Computer Visione e Deep Learning\\Progetto\\CelebAMask-HQ\\CelebAMask-HQ\\CelebA-HQ-img', 'C:\\Users\\gians_ji5genm\\OneDrive - Università Politecnica delle Marche\\UNIVPM\\Computer Visione e Deep Learning\\Progetto\\Risultati')

sg= Segmentation()
sg.process_images
