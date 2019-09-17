import sys
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

pts_ref = np.loadtxt(sys.argv[1])  # t_n+1
pts_flo = np.loadtxt(sys.argv[2])  # t_n

x, y, z = pts_flo.T
u, v, w = pts_ref.T

#cmap = cm.get_cmap('viridis', len(pts_ref)).colors
cmap = cm.get_cmap('Set1', len(pts_ref)).colors

ax.quiver(x, y, z, u-x, v-y, w-z, color=cmap, length=1., normalize=False)
ax.scatter(x, y, z, color=cmap, marker="o")
ax.scatter(u, v, w, color=cmap, marker="s")

# ax.text(x[label], y[label], z[label], range(len(z)))

plt.show()
