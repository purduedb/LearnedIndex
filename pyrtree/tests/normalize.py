import pandas as pd
import numpy as np
from sklearn import preprocessing
import csv
import argparse

parser = argparse.ArgumentParser(description="Normalize Dataset for NN")
parser.add_argument('-i', '--input', type=str,
                    default='NNdataset.csv', help='Input CSV File')
parser.add_argument('-o', '--output', type=str,
                    default='NNdisjointdataset.csv', help='Output CSV File')

args = parser.parse_args()

rows = ['ptx', 'pty', 'a']
INPUT = args.input
OUTPUT = args.output

dataframe = pd.read_csv(INPUT, header=None,
                error_bad_lines=False, delimiter=',', names=rows)

pts = dataframe.loc[:,'ptx':'pty'].values#.as_matrix()
new_dim = pts[:, 0]*pts[:, 0] + pts[:, 1]*pts[:, 1]
pts = np.column_stack((pts, pts[:,0]*pts[:,0], pts[:,1]*pts[:,1], pts[:,0]*pts[:,1], new_dim))
pts = preprocessing.scale(pts)

paths = dataframe.loc[:, 'a'].values#as_matrix()
paths = paths.astype(int)
print(paths)

with open(OUTPUT, mode='w') as dset:
    dset_writer = csv.writer(dset, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
    for idx, pt in enumerate(pts):
        val = pts[idx].tolist()
        val = val + [paths[idx],]
        dset_writer.writerow(val)

quit()
