# 🎨 Armocromia Analysis Repository

This repository contains scripts and notebooks used for armocromia (color analysis) through image processing.

---

## 📚 Contents

### 1. **Seasonal Armocromia Model (4 Classes)**
- **Notebook**: [`Fine_tuning_4_Seasons.ipynb`](Fine_tuning_4_Seasons.ipynb)  
- **Description**: This notebook includes the machine learning model trained on the dataset  
  [`/data/dataset/Armocromia/ARMOCROMIA 2. TEST_TRAIN`](data/dataset/Armocromia/ARMOCROMIA%202.%20TEST_TRAIN),  
  with images labeled according to the 4 seasons of armocromia.
- **Trained Model**: [`model_fine_tuning/modello 4 classi/armocromia_4_seasons_resnet50_full.pth`](model_fine_tuning/modello%204%20classi/armocromia_4_seasons_resnet50_full.pth)

---

### 2. **Subcategory Armocromia Model (12 Classes)**
- **Notebook**: [`Fine_tuning_12_Seasons.ipynb`](Fine_tuning_12_Seasons.ipynb)  
- **Description**: This notebook contains the machine learning model trained on the same dataset  
  [`/data/dataset/Armocromia/ARMOCROMIA 2. TEST_TRAIN`](data/dataset/Armocromia/ARMOCROMIA%202.%20TEST_TRAIN),  
  but with images labeled into 12 armocromia subcategories.
- **Trained Model**: [`model_fine_tuning/modello 12 classi/armocromia_12_seasons_resnet50_full.pth`](model_fine_tuning/modello%2012%20classi/armocromia_12_seasons_resnet50_full.pth)

---

### 3. **Facial Color Analysis**
- **Notebook**: [`Clustering.ipynb`](Clustering.ipynb)  
- **Description**: This notebook processes face images using:
  - YOLOv8 for face detection
  - Segmentation of detected faces
  - Extraction of top 3 dominant colors using K-Means (3 clusters)
  - Color clustering into 12 groups
  - Visualizations of the clusters with plotted points representing individual images

---

## 🧪 Notebook Details

- **Usage**: Run the notebooks sequentially to replicate the analysis results.  
  If you do not use the provided dataset, make sure to specify your input data paths.
- **Results**: Analysis results, including charts and visualizations, are generated within each notebook.

---

## 📝 Additional Notes

- Each notebook contains detailed instructions and code comments to guide you through the analysis process.
- Be sure to adapt model parameters and procedures to your specific needs and data structure.

---

## ⚙️ Installation Requirements

Make sure to install the following packages before running the notebooks:

```bash
%pip uninstall ultralytics
%pip install ultralytics
%pip install pyfacer
%pip install timm
%pip install scikit-learn
%pip install Pillow
%pip install plotly
