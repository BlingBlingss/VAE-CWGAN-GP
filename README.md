# VAE-CWGAN-GP
Chiller Fault Diagnosis based on VAE Enabled Generative Adversarial Networks<br>
## 1.Dependencies
In order to run the source code, the following main packages are required to
be installed in a 64 bit computer (Windows or Linux ):<br>
* tensorflow<br>
* sklearn<br>
* numpy<br>
## 2.File Discription
1. Main.py: Entry of the program.<br>
2. CWAGN_new.py: Use CWGAN-GP to generate large amounts of data.<br>
3. Diagnosis_original.py: Use a small amount of original real data as the training set and a large amount of real data as the test set to test the classification accuracy and recall rate.<br>
4. Diagnosis_Ensemble.py: After using Ensemble algorithm to select high-quality generated samples from a large amount of generated data, the high-quality generated samples are used as training data, and a large amount of real data is used as the test set to test the classification accuracy and recall rate.<br>
5. Diagnosis_vae_od.py: After using the VAE algorithm to select high-quality generated samples from a large amount of generated data, the high-quality generated samples are used as training data, and a large amount of real data is used as the test set to test the classification accuracy and recall rate.<br>
6. options.py: Experimental parameter selection, including fault level level_num, the actual number of fault used in each category select_number.<br>
7. test.py: Summary of experimental results, the experimental results after running Main.py 5 times are averaged.<br>
8. Level1.mat: Real fault data with level 1.<br>
9. Level2.mat: Real fault data with level 2.<br>
10. Level3.mat: Real fault data with level 3.<br>
11. Level4.mat: Real fault data with level 4.<br>
## 3.Run Experiments
01. Run the Main.py 5 times.<br>
02. Run the test.py to get the experimental results.<br>




