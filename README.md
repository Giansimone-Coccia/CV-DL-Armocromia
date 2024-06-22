# Repository di Analisi dell'Armocromia

Questo repository contiene script e notebook utilizzati per l'analisi dell'armocromia attraverso l'elaborazione di immagini.

## Contenuti

1. **Modello per le Stagioni dell'Armocromia (4 classi)**
   - File: [`Fine_tuning_4_Seasons.ipynb`](Fine_tuning_4_Seasons.ipynb)
   - Descrizione: Questo notebook contiene il modello di machine learning allenato sul dataset [`/data/dataset/Armocromia/ARMOCROMIA 2. TEST_TRAIN`](data/dataset/Armocromia/ARMOCROMIA%202.%20TEST_TRAIN) con immagini etichettate in base alle stagioni dell'armocromia (4 classi).
   - Modello Allenato: [`model_fine_tuning/modello 4 classi/armocromia_4_seasons_resnet50_full.pth`](model_fine_tuning/modello%204%20classi/armocromia_4_seasons_resnet50_full.pth)

2. **Modello per le Sottocategorie dell'Armocromia (12 classi)**
   - File: [`Fine_tuning_12_Seasons.ipynb`](Fine_tuning_12_Seasons.ipynb)
   - Descrizione: Questo notebook contiene il modello di machine learning allenato sul dataset [`/data/dataset/Armocromia/ARMOCROMIA 2. TEST_TRAIN`](data/dataset/Armocromia/ARMOCROMIA%202.%20TEST_TRAIN) con immagini etichettate in base alle sottocategorie dell'armocromia (12 classi).
   - Modello Allenato: [`model_fine_tuning/modello 12 classi/armocromia_12_seasons_resnet50_full.pth`](model_fine_tuning/modello%2012%20classi/armocromia_12_seasons_resnet50_full.pth)

3. **Analisi dei Colori dei Volti**
   - File: [`Clustering.ipynb`](Clustering.ipynb)
   - Descrizione: Questo notebook esegue vari processi sull'immagine dei volti utilizzando YOLOv8 per la rilevazione dei volti, segmentazione dei volti rilevati, estrazione dei 3 colori principali per ogni segmento tramite K-Means (3 cluster) e clusterizzazione dei colori dominanti (12 cluster) con successive visualizzazione dei cluster con punti che rappresentano le singole immagini.

## Dettagli sui Notebook

- **Requisiti**: Assicurarsi di avere tutti i moduli Python necessari installati, inclusi TensorFlow, OpenCV, e scikit-learn.
- **Utilizzo**: Eseguire i notebook in sequenza per replicare i risultati dell'analisi. Se non si vuole riutilizzare i dataset forniti, i file di dati di input devono essere specificati nuovamente.
- **Risultati**: I risultati dell'analisi, inclusi grafici e visualizzazioni, sono generati all'interno dei notebook stessi.

## Note Aggiuntive

- Ogni notebook contiene istruzioni dettagliate e commenti nel codice per guidare attraverso il processo di analisi.
- Assicurarsi di adattare i parametri del modello e dei processi secondo le proprie necessità e specifiche dei dati.

## Requisiti di Installazione

Assicurarsi di installare i seguenti pacchetti prima di eseguire i notebook:

```bash
%pip uninstall ultralytics
%pip install ultralytics
%pip install pyfacer
%pip install timm
%pip install scikit-learn
%pip install Pillow
%pip install plotly
