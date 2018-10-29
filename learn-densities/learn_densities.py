import numpy as np
import seaborn as sns
import pickle

from get_densities import *
from generate_model import *



if __name__=='__main__':
    
  files = sorted(glob.glob('/scratch/group/kalescky/projects/01_ml_dft/01_b3lyp_vdz/structures/*/*.cube', recursive = True))

  atomic_numbers, locations, densities = get_cube_data(files[0])
  # print(raw_atomic_data)

  # atomic_data = [get_cube_data(data_file) for data_file in files]
  # input_data = [[] for atom in atomic_data]

  atomic_data = clean_density_data(atomic_numbers, locations, densities)
  # print(atomic_data.shape, atomic_data)
  # pickle.dump(atomic_data, open('atomic_data_clean.p', 'wb'))


  model = generate_model(atomic_data)
  model.compile(optimizer='sgd',
                loss='mean_squared_error',
                metrics=['acc'])
  history_train = model.fit(
    atomic_data[]
