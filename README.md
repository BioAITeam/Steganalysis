# Steganalysis


GBRAS-Net: A Deep LearningArchitecture For Spatial ImageSteganalysis


<p align="justify">Advances in deep learning have provided alternative approaches to a variety of complexproblems, such as in the domain of spatial image steganalysis using convolutional neural networks (CNN),where several CNN architectures and datasets have been developed in the last few years. In this work wepresent a novel CNN architecture that improves the current state-of-the-art in a robust manner. We usedthe BOSSbase 1.01 dataset with experimental setups including the adaptive steganography algorithms S-UNIWARD and WOW with payloads of 0.2 and 0.4 bits per pixel. As results are obtained, it is more complex to improve them. In fact, some researchershave no longer proposed other architectures but only improvements with databases for example. However,we propose a new convolutional neural network that manages to become the new state-of-the-art. This articleintroduces the previous architectures used in this classification problem. All concepts covered at the time ofits design and creation are included. The experiments developed are carried out mainly with the BOSSbase1.01 database. Improvements are presented in the adaptive steganography algorithms S-UNIWARD andWOW with payloads of 0.2 and 0.4 bits per pixel. By the time it has 0.2 bits per pixel, the improvementsfor the aforementioned algorithms are presented, including the increase of the database with BOWS2 fortraining. In addition, the performance of the convolutional neural network for the HILL, MiPOD and HUGOsteganography algorithms is shown. The architecture is more accurate than all the previous ones for allthe mentioned adaptive steganography methods. The present GBRAS-Net architecture is developed usingTensorFlow, and with this we want it to be relatively easy to reproduce by researchers in the field. We alsopresent a methodology to obtain a new database to use in this domain. Due to this, a step forward is takenin the spatial image steganalysis domain.</p>


<strong>GBRAS-Net.ipynb</strong>: Contains the architecture and codes needed to reproduce the results.


<strong>SRM_Kernels1.npy</strong>: It contains the values of the 30 filters (SRM) needed for the first layer of the proposed architecture.


<strong>TIF2PGM.m</strong>: Allows you to convert images from the Alaska database from TIFF to PGM format.


<strong>model.png</strong>: GBRAS-Net Architecture.

The data set used to reproduce the results can be downloaded from: <a href="https://drive.google.com/drive/folders/1G5vdhW11_qKfVC6W8_pfJpstVkXUk1QQ?usp=sharing">Here</a>
