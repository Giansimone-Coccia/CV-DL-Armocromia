from src.face_detection import FaceDetection
from src.segmentation import Segmentation


fd = FaceDetection()
fd.face_detection()

sg= Segmentation()
sg.extract_dominant_colors(sg.process_images())
