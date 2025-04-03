import numpy as np

theta = np.radians(30)
R = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])
C = np.array([0, 0, -10])
t = -R @ C
Rt = np.column_stack((R, t))
print("R =\n", R)
print("t =\n", t)
print("[R|t] =\n", Rt)
