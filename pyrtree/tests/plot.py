import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = ['ptx', 'pty', 'a', 'b', 'c', 'd', 'e']
dataframe = pd.read_csv("datasetGauss.csv", header=None, error_bad_lines = False,
                        delimiter=':', names=rows)

pts = dataframe.loc[:, 'ptx':'pty'].values
paths = dataframe.loc[:, 'a':].values
paths = paths.astype(int)
#paths = paths[:, 0]
pts2 = []
paths2 = []
for i, path in enumerate( paths):
    paths2.append(path[0])
#    if path[0] == 0:
#        pts2.append(pts[i])
#        paths2.append(path[1])
#    else:
#        print(path)

paths = paths2
color = ['r', 'g', 'b', 'c', 'm', 'y']
colors = []

for i in paths:
    colors.append(color[i])

print(len(colors), len(pts))
#pts2 = np.array(pts2)
#pts = pts2
plt.scatter(pts[:,0], pts[:,1],s=5.5, alpha=0.3, color=colors)
plt.title("Level 1 Node 0")
plt.savefig("Root_node.png")

