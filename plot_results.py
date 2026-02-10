import openmc
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

# Load statepoint file
sp = openmc.StatePoint("statepoint.100.h5")

# Plot 2D mesh flux tally
mesh_tally = sp.get_tally(name='mesh flux tally')

mesh = mesh_tally.find_filter(openmc.MeshFilter).mesh
flux = mesh_tally.get_slice(scores=['flux']).mean.flatten()

# Reshape flux data to match mesh dimensions
flux_reshaped = flux.reshape(mesh.dimension[1], mesh.dimension[0])  # Note the order for plotting

# Divide by volume to convert to flux
flux_reshaped = flux_reshaped/mesh.volumes.reshape(mesh.dimension[1], mesh.dimension[0])

flux_log_norm = colors.LogNorm(vmin=1, vmax=1e6)

x = np.linspace(mesh.lower_left[0], mesh.upper_right[0], mesh.dimension[0] + 1)
y = np.linspace(mesh.lower_left[1], mesh.upper_right[1], mesh.dimension[1] + 1)

plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(8, 10))
c = ax.pcolormesh(x, y, flux_reshaped, shading='auto', cmap='magma', norm=flux_log_norm)
ax.set_aspect('equal')
fig.colorbar(c, ax=ax, label='Neutron Flux (n/cm²-s)')
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_title('2D Neutron Flux Distribution')
fig.tight_layout()
fig.savefig('XY_neutron_flux.png', dpi=300)

# Mesh dose tally
dose_tally = sp.get_tally(name='mesh neutron dose tally')
dose = dose_tally.get_slice(scores=['flux']).mean.flatten()

dose_reshaped = dose.reshape(mesh.dimension[1], mesh.dimension[0])  # Note the order for plotting

# Convert to mrem/hr
dose_reshaped = dose_reshaped/mesh.volumes.reshape(mesh.dimension[1], mesh.dimension[0]) * 3600 / 10000000 # from pSv/s to mrem/hr
#log_dose = np.log10(dose_reshaped + 1e-20)  # Avoid log(0) issues

dose_log_norm = colors.LogNorm(vmin=0.1, vmax=1000)

dose_fig, dose_ax = plt.subplots(figsize=(8, 10))
dose_c = dose_ax.pcolormesh(x, y, dose_reshaped, shading='auto', cmap='magma', norm=dose_log_norm)
contours = dose_ax.contour(x[:-1], y[:-1], dose_reshaped, levels=[3, 10, 100], colors='white')
dose_ax.clabel(contours, inline=True, fontsize=18, inline_spacing=10)            
dose_ax.set_aspect('equal')
dose_fig.colorbar(dose_c, ax=dose_ax, label='Neutron Dose Rate (mrem/hr)')
dose_ax.set_xlabel('X (cm)')
dose_ax.set_ylabel('Y (cm)')
dose_ax.set_title('2D Neutron Dose Rate Distribution')
dose_fig.tight_layout()

dose_fig.savefig('XY_neutron_dose_rate.png', dpi=300)

# Plot detector tally
det_volume = (np.pi * (1.5)**2) * 1  # cm^3

detector_tally = sp.get_tally(name='detector tally')
detector_flux = detector_tally.get_slice(scores=['flux']).mean.flatten()
detector_flux = detector_flux / det_volume  # Assuming detector volume is 1

det_spec_fig, det_spec_ax = plt.subplots(figsize=(8,6))
energy_bins = openmc.mgxs.GROUP_STRUCTURES['CCFE-709']
bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])
det_spec_ax.step(bin_centers, detector_flux, where='mid')
det_spec_ax.set_xscale('log')
det_spec_ax.set_yscale('log')
det_spec_ax.set_xlabel('Energy (eV)')
det_spec_ax.set_ylabel('Neutron Flux (n/cm²-s-eV)')
det_spec_ax.set_title('Neutron Energy Spectrum at Detector')
det_spec_fig.tight_layout()
det_spec_fig.savefig('detector_neutron_spectrum.png', dpi=300)

plt.show()

sp.close()


