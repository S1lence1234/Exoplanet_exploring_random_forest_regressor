import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score as err
from sklearn.model_selection import train_test_split as spl
import numpy as np
dataset = pd.read_csv('./train.csv')

for col in dataset.columns:
   dataset.loc[dataset[col] == 'poor', col] = 1
   dataset.loc[dataset[col] == 'fair', col] = 2
   dataset.loc[dataset[col] == 'good', col] = 3
   dataset.loc[dataset[col] == 'main_sequence_star', col] = 1
   dataset.loc[dataset[col] == 'white_dwarf', col] = 2
   dataset.loc[dataset[col] == 'quasar', col] = 3
   dataset.loc[dataset[col] == 'red_giant', col] = 4
   dataset.loc[dataset[col] == 'exoplanet_candidate', col] = 5
   dataset.loc[dataset[col] == 'galaxy', col] = 6

decoder = {1:'main_sequence_star', 2:'white_dwarf', 3:'quasar', 4:'red_giant', 5:'exoplanet_candidate', 6:'galaxy'}

dataset_features = ['u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag', 'u_g', 'g_r', 'r_i',
       'i_z', 'parallax', 'pm_ra', 'pm_dec', 'radial_velocity', 'snr_u',
       'snr_g', 'snr_r', 'snr_i', 'snr_z', 'obs_count', 'cloud_factor',
       'background_noise', 'h_alpha_strength', 'oIII_strength',
       'na_d_strength', 'extinction', 'quality_flag']

X = dataset[dataset_features]

y = dataset['type']

train_x, val_x, train_y, val_y = spl(X, y, random_state=1)
train_x, val_x, train_y, val_y = spl(train_x, train_y, random_state=0)

model = RandomForestRegressor(random_state=3)

model.fit(train_x, train_y)

decision = model.predict(val_x)

np.around(decision, 0)

train_exo = 0
val_exo = 0

for x in train_y.values:
   if x == 5:
      train_exo+= 1
for x in val_y.values:
   if x == 5:
      val_exo += 1
print('Exo in train dataset:', train_exo)
print('Exo in test dataset:', val_exo)
print('Accuracy: ', round(err(decision.astype(int), val_y.values.astype(int))*100, 1), '%')

print('Exoplanet prediction')
for a in range(0, len(decision)):
   x = decoder[round(decision.tolist()[a])]
   y = decoder[val_y.tolist()[a]]
   if y == 'exoplanet_candidate':
      if x == y:
         print(a,':', 'found')
      else:
         print(a, ":", 'lost') 