import time
start_time = time.time()
# import os
import CWGAN_new
import Diagnosis_original
import Diagnosis_Ensemble
import Diagnosis_vae_od
import Detection_original
import Detection_Ensemble
import Detection_vae_od
end_time = time.time()
total_time = end_time - start_time
print(total_time)
