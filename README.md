# Sementic-Segmentation on LIVECell Dataset 
---
### LIVECell Dataset (https://sartorius-research.github.io/LIVECell/)
LIVECell is a large, high-quality, manually annotated and expert-validated dataset of phase-contrast images, consisting of over 1.6 million cells from a diverse set of cell morphologies and culture densities.
<img src="https://production-media.paperswithcode.com/datasets/cell-example.png" width="640" height="300" />

---
### Model used : CANet with few modifications.
* Loss Used : dice_loss + (1 * focal_loss)  
* Total Epochs : 10  
* Training IOU Score : 0.6533  
* Validation IOU Score : 0.7483  
* Training F1 Score : 0.7812  
* Validation F1 Score : 0.8532  

---
### Model Performance.
<img src="/performace_curves.JPG" width="660" height="280" />

---
### Visualization of model prediction.
<img src="/model_prediction.JPG" width="680" height="240" />

---
### Reference :
  1. https://arxiv.org/abs/1903.02351
  2. https://www.nature.com/articles/s41592-021-01249-6
  3. https://www.sciencedirect.com/science/article/abs/pii/S0262885621002146


