# Steganalysis


<p align="center"><strong>GBRAS-Net: A Deep Learning Architecture For Spatial Image Steganalysis</strong></p>


<p align="justify">Advances in Deep Learning (DL) have provided alternative approaches to various complex problems, including the domain of spatial image steganalysis using Convolutional Neural Networks (CNN). Several CNN architectures have been developed in the last few years improving steganographic images' detection accuracy. This work presents a novel CNN architecture which involves a preprocessing stage using filter banks to enhance steganographic noise, a feature extraction stage with convolutional layers and its depthwise and separable variations, and skip connections. The CNN performance evaluation used the BOSSbase 1.01 dataset with experimental setups, including the adaptive steganographic algorithms S-UNIWARD and WOW with payloads of 0.2 and 0.4 bits per pixel (bpp). The results achieved outperform those reported in the literature in every experimental setting. This work contributes to the classification accuracy presenting improvements, for example, in WOW and S-UNIWARD on 0.4bpp. (an accuracy value of 89.8% and 87.1%, respectively). Compared with the state-of-the-art, the proposed CNN has improvements of accuracy on the BOSSbase $1.01$ dataset of 1.7%  and 2.6%, respectively, for the two algorithms mentioned.</p>

## Folders

  -**Matlab**: Allows you to convert images from the Alaska database from TIFF to PGM format.
  
  
  -**Trained_Models**: Best trained models.
  
  
  -**images**: Some GBRAS-Net results.
  

## Files

  -**GBRAS-Net.ipynb**: Contains the architecture and codes needed to reproduce the results.
  
  
  -**SRM_Kernels1.npy**: It contains the values of the 30 filters (SRM) needed for the first layer of the proposed architecture.
  
  
  -**model.png**: GBRAS-Net Architecture.
  
  
  -**Trained_Models.ipynb**: Load the best models and reproduce results.
  
  
  -**Metrics_Visualization.ipynb**: loads the WOW 0.4 bpp model and the database, generates classification report, confusion matrix and ROC curves and class prediction error.
  
  
  -**MV.py**: Necessary libraries for the results visualization.
  
  
  -**Requirements.txt**: File containing the libraries needed to reproduce the results of GBRAS-Net.
  
  
## Databases

The data set used to reproduce the results can be downloaded from: <a href="https://drive.google.com/drive/folders/1G5vdhW11_qKfVC6W8_pfJpstVkXUk1QQ?usp=sharing">Here</a>. Images taken from: <a href="http://agents.fel.cvut.cz/boss/index.php?mode=VIEW&tmpl=materials">BOSS competition</a>, <a href="http://bows2.ec-lille.fr/index.php?mode=VIEW&tmpl=index1">BOWS2</a> and <a href="https://alaska.utt.fr/">ALASKA2</a> .

## GBRAS-NET Architecture
![GBRAS-Net Architecture](https://github.com/BioAITeam/Steganalysis/blob/main/model.pdf?raw=true "GBRAS-Net Architecture")
