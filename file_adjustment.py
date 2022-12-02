import numpy as np
import glob
import os

files = glob.glob("/Users/patricknaylor/Desktop/Field_Detection/Images/Masked/*.csv")
for file in files:
    arr = np.loadtxt(file, delimiter=",", dtype="int8")
    name = file[:-4] + ".npy"
    np.save(name, arr)
    os.remove(file)
