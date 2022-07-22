#%%
from windrose import WindroseAxes
import matplotlib.pyplot as plt

theta = [0, 30]
theta = [360 - x for x in theta] # Take the opposite angle
speed = [10, 8]

ax = WindroseAxes.from_ax()
ax.bar(theta, speed)
plt.show()
# %%
