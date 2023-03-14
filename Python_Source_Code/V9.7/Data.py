import numpy as np


def read_data(file_path):
    """
    Reads the data from the Data.csv file and creates the vectors accordingly.
    """
    f = np.genfromtxt(file_path, delimiter=',')
    Eload = f[:, 0]
    G = f[:, 1]
    T = f[:, 2]
    Vw = f[:, 3]
    return Eload, G, T, Vw
