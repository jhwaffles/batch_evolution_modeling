import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
import sklearn.model_selection as skm

###Prepare Data
df=pd.read_excel("synthesized_data.xlsx")

#filter batches
df_filtered = df.query('solution_type == 1').query('is_synthetic==True').copy()

#define meta columns
meta_cols=['batch_id','batch_time_h']
#define y column
y_col=['batch_time_h']
y=df_filtered.filter(y_col)
print("y shape:", y.shape)

#define columns
X_cols=['acetate_mM', 'glucose_g_L', 'mg_mM','nh3_mM', 'phosphate_mM']
df_model = df_filtered[meta_cols+X_cols].copy()
df_model=df_model.sort_values(['batch_id','batch_time_h'])
df_model.head()

###run PLS, choose components

X=np.array(df_model[X_cols])
print("X shape:", X.shape)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

K = 5
kfold = skm.KFold(K,random_state=0,shuffle=True)

pls = PLSRegression()

param_grid = {'n_components':range(1, len(X_cols)+1)}
grid = skm.GridSearchCV(pls ,param_grid ,cv=kfold, scoring='neg_mean_squared_error')
grid.fit(X_scaled, y)

# Extract results
mean_mse = -grid.cv_results_['mean_test_score']
std_mse = grid.cv_results_['std_test_score'] / np.sqrt(K)
optimal_n_components = grid.best_params_['n_components']

# Plot Cross-Validated MSE vs. # of PLS Components
plt.figure(figsize=(8, 8))
plt.errorbar(param_grid['n_components'], mean_mse, yerr=std_mse, fmt='o-', capsize=5)
plt.axvline(optimal_n_components, color='r', linestyle='--', label=f'Optimal: {optimal_n_components}')
plt.ylabel("Cross-validated MSE", fontsize=16)
plt.xlabel("# of PLS Components", fontsize=16)
plt.xticks(list(param_grid['n_components'])[::2])  # Show every other tick
plt.title("Optimal Number of PLS Components", fontsize=18)
plt.legend()
plt.show()

###once n components is chosen, get T matrix, scores
n_comp = 3
pls_final = PLSRegression(n_comp)
pls_final.fit(X_scaled, y)

T = pls_final.transform(X_scaled) 
print("T shape:", T.shape)

score_cols = [f"comp_{i+1}" for i in range(n_comp)]
T_df = pd.DataFrame(T, columns=score_cols, index=df_model.index)

print("complete")
###Transfor matrix so we can plot the components over time.
#Transform T from (NxJ)xA to Nx(AxJ)
# N=17
# J=25
# A=3
# T_3D=T.reshape(N,J,A,order='C')
# print(T_3D.shape)

# # New shape: N x A x J
# T_permuted = T_3D.transpose(0, 2, 1)
# print(T_permuted.shape)

# X_T = T_permuted.reshape(N, A * J, order='C')
# print(X_T.shape)
# # X_T shape: N x (A*J)

