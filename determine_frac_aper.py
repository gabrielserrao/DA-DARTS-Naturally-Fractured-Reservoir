#%%
import numpy as np
import meshio
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

#%%
def calc_aperture(sigma_n):
    JRC = 7.225
    JCS = 17.5
    e_0 = 0.15
    v_m = -0.1032 - 0.0074 * JRC + 1.135 * (JCS / e_0) ** -0.251
    K_ni = -7.15 + 1.75 * JRC + 0.02 * JCS / e_0
    return e_0 - (1 / v_m + K_ni / sigma_n) ** -1


DIR = 'mesh_files'
scale = 15
mesh_file = os.path.join(DIR, 'mesh_{:}_real_6.msh'.format(scale))
mesh_data = meshio.read(mesh_file)
num_frac = mesh_data.cells[0].data.shape[0]
act_frac_sys = np.zeros((num_frac, 4))

for ii in range(num_frac):
    ith_line = mesh_data.points[mesh_data.cells[0].data[ii][:2]]
    act_frac_sys[ii, :2] = ith_line[0, :2]
    act_frac_sys[ii, 2:] = ith_line[1, :2]

# Plot to check if it worked:
#plt.figure()
#plt.plot(np.array([act_frac_sys[:, 0], act_frac_sys[:, 2]]),
#         np.array([act_frac_sys[:, 1], act_frac_sys[:, 3]]), color='black')
#plt.show()

epsilon = 1e-4
dx = act_frac_sys[:, 0] - act_frac_sys[:, 2] + epsilon * np.random.rand(num_frac)
dy = act_frac_sys[:, 1] - act_frac_sys[:, 3] + epsilon * np.random.rand(num_frac)
rotation = 0
angles = np.arctan(dy / dx) * 180 / np.pi + rotation + epsilon * np.random.rand(num_frac)
sigma_H = 5
sigma_h = 1
sigma_n = (sigma_H + sigma_h) / 2 + (sigma_H + sigma_h) / 2 * np.cos(angles * np.pi / 180 * 2)
factor_aper = 2.5
fracture_aper = calc_aperture(sigma_n) * 1e-3 * factor_aper
fracture_aper[fracture_aper < 1e-9] = 1e-9
fracture_aper[fracture_aper > 0.15 * 1e-3 * factor_aper] = 0.15 * 1e-3 * factor_aper

m_to_mm = 1e3
min_aper = 1e-9 * m_to_mm
max_aper = 0.15 * 1e-3 * factor_aper * m_to_mm
fracs = fracture_aper * m_to_mm
fracs[fracs < min_aper] = min_aper
fracs[fracs > max_aper] = max_aper
norm = colors.Normalize(min_aper, max_aper)
colors_aper = cm.viridis(norm(fracs))
cutoff_value = min_aper

fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=400, facecolor='w', edgecolor='k')
for ii in range(len(fracs)):
    if fracs[ii] >= cutoff_value:
        ax.plot(np.array([act_frac_sys[ii, 0], act_frac_sys[ii, 2]]),
                np.array([act_frac_sys[ii, 1], act_frac_sys[ii, 3]]),
                 color=colors_aper[ii, :-1])

ax.axis('equal')

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)

sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm._A = []
cbar = plt.colorbar(sm)
cbar.set_label(r'Aperture, $a$ [mm]')

#plt.tight_layout()
#plt.savefig("variable_aperture_base_case_scale_{:}.pdf".format(scale))
#plt.show()

# %%
