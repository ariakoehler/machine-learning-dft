#!/bin/env python
import glob, re
import itertools as itt
import numpy as np
import matplotlib as plt

def get_cube_data(cube_file):
  with open(cube_file, 'r') as f:
    filedata = f.read()
  datasplit = filedata.splitlines()
  n = int(datasplit[2].split()[0])
  dims = (int(datasplit[3].split()[0]), int(datasplit[4].split()[0]), int(datasplit[5].split()[0]))
  atomic_numbers = []
  coordinates = []
  for l in datasplit[6:6+n]:
    l = l.split()
    atomic_numbers.append(int(l[0]))
    coordinates.append((float(l[2]), float(l[3]), float(l[4])))
  datalines = iter(datasplit)
  p_scinot = re.compile("-?\\d\\.\\d*[de][+-]\\d+", re.X | re.I)
  dataiter = itt.chain.from_iterable(p_scinot.finditer(l) for l in datalines)
  density = np.array(list(map(lambda s: np.float(s.group(0)), dataiter))).reshape(dims)
  return (atomic_numbers, coordinates, density)


# def get_similarity_features():
#   return


def clean_density_data(atomic_numbers, locations, densities, max_particles=6, grid_spacing=10):

  grid_spacing=min(locations)
  density_array = np.array(densities)
  out_array = np.zeros((density_array.size, max_particles * 4 + 1))
  
  iter = np.nditer(density_array, flags=['multi_index'])
  while not iter.finished:
    location = np.array(iter.multi_index) * grid_spacing
    out_array[int(np.prod(iter.multi_index)), :] = process_pixel(iter[0], location, atomic_numbers, locations)
    iter.iternext()
    
  return out_array


def process_pixel(pixel, location, atomic_numbers, atom_locations):
  pixel_data = []

  for atom, aloc in zip(atomic_numbers, atom_locations):
    pixel_data.append(atom)
    for pix_pos, atom_pos in zip(location, aloc):
      pixel_data.append(pix_pos - atom_pos)

  pixel_data.append(pixel)

  return pixel_data


if __name__=='__main__':

  files = sorted(glob.glob('/scratch/group/kalescky/projects/01_ml_dft/01_b3lyp_vdz/structures/*/*.cube', recursive = True))

  atomic_numbers, coordinates, density = get_cube_data(files[0])
  print(atomic_numbers, coordinates, density)

