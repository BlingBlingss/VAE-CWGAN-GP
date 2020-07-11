import numpy as np
level_num = 1
select_number = 10
result = np.loadtxt("./temp_result/Diagnosis_"+str(select_number)+"Level"+str(level_num)+"_Accuracy_result.txt")
result1 = np.loadtxt("./temp_result/Diagnosis_"+str(select_number)+"Level"+str(level_num)+"_f1_score_result.txt")

# Accuracy
result_original = result[[0, 3, 6, 9, 12]]
result_Ensemble = result[[1, 4, 7, 10, 13]]
result_vae_od = result[[2, 5, 8, 11, 14]]
print("#########ACCURACY:##########")
# print("result_original_mean:")
print(np.mean(result_original, 0))
# print("result_Ensemble_mean:")
print(np.mean(result_Ensemble, 0))
# print("result_vae_od_mean:")
print(np.mean(result_vae_od, 0))

# f1_score
result_original1 = result1[[0, 3, 6, 9, 12]]
result_Ensemble1 = result1[[1, 4, 7, 10, 13]]
result_vae_od1 = result1[[2, 5, 8, 11, 14]]
print("#########F1-SCORE:##########")
print(np.mean(result_original1, 0))
print(np.mean(result_Ensemble1, 0))
print(np.mean(result_vae_od1, 0))