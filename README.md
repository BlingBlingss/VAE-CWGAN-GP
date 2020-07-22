# VAE-CWGAN-GP
Chiller Fault Diagnosis based on VAE Enabled Generative Adversarial Networks.This work attacks the fact that faulty training samples are usually much harder to be collected than the normal training samples in the practice of chiller automated fault diagnosis (AFD). Modern supervised learning chiller AFD relies on sufficient number of faulty training samples to train the classifier. When the number of faulty training samples is not enough, the supervised learning based AFD methods fail to work. This study proposed a data augmentation method combining variational auto-encoder (VAE) and generative adversarial network (GAN) for generating synthetic faulty training samples to re-balance the training dataset for supervised learning based AFD methods. The proposed algorithm has been carefully implemented and practically proved to be more effective than existing methods in the literature.<br>
## 1.Dependencies
In order to run the source code, the following main packages are required to
be installed in a 64 bit computer (Windows or Linux ):<br>
* tensorflow<br>
* sklearn<br>
* numpy<br>
## 2.File Discription
01. *Main.py* :  Entry of the program.<br>
02. *CWAGN_new.py*  : Use CWGAN-GP to generate large amounts of data.<br>
03. *Diagnosis_original.py* : Use a small amount of original real data as the training set and a large amount of real data as the test set to test the classification accuracy and recall rate.<br>
04. *Diagnosis_Ensemble.py* : After using Ensemble algorithm to select high-quality generated samples from a large amount of generated data, the high-quality generated samples are used as training data, and a large amount of real data is used as the test set to test the classification accuracy and recall rate.<br>
05. *Diagnosis_vae_od.py* : After using the VAE algorithm to select high-quality generated samples from a large amount of generated data, the high-quality generated samples are used as training data, and a large amount of real data is used as the test set to test the classification accuracy and recall rate.<br>
06. *options.py*  : Experimental parameter selection, including fault level level_num, the actual number of fault used in each category select_number.<br>
07. *test.py* : Summary of experimental results, the experimental results after running Main.py 5 times are averaged.<br>
08. *Level1.mat*  : Real fault data with level 1.<br>
09. *Level2.mat*  : Real fault data with level 2.<br>
10. *Level3.mat*  : Real fault data with level 3.<br>
11. *Level4.mat*  : Real fault data with level 4.<br>
## 3.Run Experiments
01. Run the Main.py 5 times.<br>
02. Run the test.py to get the experimental results.<br>




