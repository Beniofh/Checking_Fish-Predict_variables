import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

def MetricChallangeIABiodiv(y_test_pred, y_test_true):
    y_test_true_copie=y_test_true.copy()
    y_test_pred_copie=y_test_pred.copy()

    y_test_true_copie[y_test_true_copie<1]=1
    y_test_pred_copie[y_test_pred_copie<1]=1


    sum_S = 0
    for site in range(y_test_true_copie.shape[0]):
        S=sum(abs(np.log10(y_test_pred_copie[site])-np.log10(y_test_true_copie[site])))/y_test_true_copie.shape[1]
        sum_S += S
    mean_S = sum_S/y_test_true_copie.shape[0]
    return mean_S    


# Chemin 
path_dir = "../data"
data = "Galaxy117-Sort_on_data_82_log(n+1)_vec_all"

path_dir_2 = "../data/"
data_2 = "2023-04-17_15-32-35"

path_dir_3 = "../data/"
data_3 = "Galaxy117-Sort_on_data_82_n_vec_value"

# Nombre de décimal pour le résultat
score_decimal=10

# type d'inputs pour le RF 
inputs ='env+spatial'  # 'env+spatial+cnn_output' 'env+spatial' 'cnn_output'



# récupération des données
# pour le type de vecteur
df = pd.read_csv(path_dir + "/" + data + ".csv", sep=',', index_col='SurveyID')
# pour les sorties du CNN
df_2 = pd.read_csv(path_dir_2 + "/" + data_2 + ".csv", sep=',', index_col='SurveyID')
# pour les variables env et spatiales
df_3 = pd.read_csv(path_dir_3 + "/" + data_3 + ".csv", sep=',', index_col='SurveyID')

df_3bis = df_3[['SiteLat',
            'SiteLong',
            'TCI_sentinel_band_0_sd',
            'TCI_sentinel_band_0_central_value',
            'TCI_sentinel_band_1_sd',
            'TCI_sentinel_band_1_central_value',
            'TCI_sentinel_band_2_sd',
            'TCI_sentinel_band_2_central_value',
            'bathymetry_band_0_sd',
            'bathymetry_band_0_central_value',
            'bathy_95m_band_0_sd',
            'bathy_95m_band_0_mean_15x15',
            'chlorophyll_concentration_1km_band_0_sd',
            'chlorophyll_concentration_1km_band_0_mean_30x30',
            'east_water_velocity_4_2km_mean_day_lite_band_0_sd',
            'east_water_velocity_4_2km_mean_day_lite_band_0_mean_7x7',
            'east_water_velocity_4_2km_mean_day_lite_band_1_sd',
            'east_water_velocity_4_2km_mean_day_lite_band_1_mean_7x7',
            'east_water_velocity_4_2km_mean_day_lite_band_2_sd',
            'east_water_velocity_4_2km_mean_day_lite_band_2_mean_15x15',
            'east_water_velocity_4_2km_mean_month_lite_band_0_sd',
            'east_water_velocity_4_2km_mean_month_lite_band_0_mean_7x7',
            'east_water_velocity_4_2km_mean_month_lite_band_1_sd',
            'east_water_velocity_4_2km_mean_month_lite_band_1_mean_7x7',
            'east_water_velocity_4_2km_mean_month_lite_band_2_sd',
            'east_water_velocity_4_2km_mean_month_lite_band_2_mean_15x15',
            'meditereanean_sst_band_0_sd',
            'meditereanean_sst_band_0_mean_15x15',
            'north_water_velocity_4_2km_mean_day_lite_band_0_sd',
            'north_water_velocity_4_2km_mean_day_lite_band_0_mean_7x7',
            'north_water_velocity_4_2km_mean_day_lite_band_1_sd',
            'north_water_velocity_4_2km_mean_day_lite_band_1_mean_7x7',
            'north_water_velocity_4_2km_mean_day_lite_band_2_sd',
            'north_water_velocity_4_2km_mean_day_lite_band_2_mean_15x15',
            'north_water_velocity_4_2km_mean_month_lite_band_0_sd',
            'north_water_velocity_4_2km_mean_month_lite_band_0_mean_7x7',
            'north_water_velocity_4_2km_mean_month_lite_band_1_sd',
            'north_water_velocity_4_2km_mean_month_lite_band_1_mean_7x7',
            'north_water_velocity_4_2km_mean_month_lite_band_2_sd',
            'north_water_velocity_4_2km_mean_month_lite_band_2_mean_15x15',
            'salinity_4_2km_mean_day_lite_band_0_sd',
            'salinity_4_2km_mean_day_lite_band_0_mean_7x7',
            'salinity_4_2km_mean_day_lite_band_1_sd',
            'salinity_4_2km_mean_day_lite_band_1_mean_7x7',
            'salinity_4_2km_mean_day_lite_band_2_sd',
            'salinity_4_2km_mean_day_lite_band_2_mean_15x15',
            'salinity_4_2km_mean_month_lite_band_0_sd',
            'salinity_4_2km_mean_month_lite_band_0_mean_7x7',
            'salinity_4_2km_mean_month_lite_band_1_sd',
            'salinity_4_2km_mean_month_lite_band_1_mean_7x7',
            'salinity_4_2km_mean_month_lite_band_2_sd',
            'salinity_4_2km_mean_month_lite_band_2_mean_15x15',
            'sea_water_potential_temperature_at_sea_floor_4_2km_mean_day_band_0_sd',
            'sea_water_potential_temperature_at_sea_floor_4_2km_mean_day_band_0_mean_7x7',
            'sea_water_potential_temperature_at_sea_floor_4_2km_mean_month_band_0_sd',
            'sea_water_potential_temperature_at_sea_floor_4_2km_mean_month_band_0_mean_7x7',]]

if inputs == 'env+spatial+cnn_output' :
    df_var =  pd.merge(df_2, df_3bis, left_index=True, right_index=True)
elif inputs == 'env+spatial' :
    df_var =  pd.merge(df_2.subset, df_3bis, left_index=True, right_index=True)
elif inputs == 'cnn_output' : 
    df_var = df_2
'''
code pour rechercher l'emplacement des NaN :'
nan_locations = np.where(df_var.isna()) #  le premier tableau contenant les indices des lignes et le deuxième tableau contenant les indices des colonnes où se trouvent les valeurs NaN dans le dataframe
df_var.iloc[:, 209].name
'''

# split des données
train_set_index = df_var.index[df_var.subset=='train']
val_set_index = df_var.index[df_var.subset=='val']

# nom de la collone du vecteur d'espèce
name_col_sp = df.columns[-1]


### Mise en place du train set
n=0
for j in train_set_index :
    if n==0:
        X = np.array([df_var.loc[j][1:]])        
        y = np.array([eval(df.loc[j,name_col_sp])])
    else :
        X=np.append(X,
                    np.array([df_var.loc[j][1:]]),
                    axis=0)
        y=np.append(y, np.array([eval(df.loc[j,name_col_sp])]), axis=0) 
    n=n+1
    
### Mise en place du test set        
n=0
for i in val_set_index : 
    if n==0:
        X_test = np.array([df_var.loc[i][1:]])        
        y_test_true = np.array([eval(df.loc[i,name_col_sp])])
    else :
        X_test = np.append(X_test ,
                                np.array([df_var.loc[i][1:]]),
                                axis=0)
        y_test_true = np.append(y_test_true , np.array([eval(df.loc[i,name_col_sp])]), axis=0) 
    n=n+1
    
### Configuration du l'entrainement
regr = RandomForestRegressor(max_depth=2, random_state=0, criterion="squared_error") 

### Entrainement
regr.fit(X, y)

### Résultats
y_test_pred=regr.predict(X_test)


MAE = round(np.mean(mean_absolute_error(y_test_true, y_test_pred,multioutput="raw_values")),score_decimal)
print(f'mean_absolute_error: {MAE}')

MSLE = round(np.mean(mean_squared_log_error(y_test_true, y_test_pred,multioutput="raw_values")),score_decimal)
print(f'mean_squared_log_error: {MSLE}')

MSE = round(np.mean(mean_squared_error(y_test_true, y_test_pred,multioutput="raw_values")),score_decimal)
print(f'mean_squared_error: {MSE}')

metric_challange_ia_biodiv = round(MetricChallangeIABiodiv(np.exp(y_test_pred)-1, np.exp(y_test_true)-1),score_decimal)
print(f'metric_challange_ia_biodiv: {metric_challange_ia_biodiv}')


R2_sites = round(r2_score(np.transpose(y_test_true), np.transpose(y_test_pred),multioutput="uniform_average"),score_decimal)
print(f'R2_sites: {R2_sites}')

#R2_sites_log = round(r2_score(np.transpose(np.log(y_test_true+1)), np.transpose(np.log(y_test_pred+1)),multioutput="uniform_average"),3)
#print(f'R2_sites_of_log(n+1): {R2_sites_log}')


    

X.shape
X_test.shape

