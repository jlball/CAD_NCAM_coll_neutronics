import openmc
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot OpenMC simulation results.')
parser.add_argument("directory", help="Directory containing OpenMC statepoint file")
args = parser.parse_args()

# Load statepoint file
sp = openmc.StatePoint(f"{args.directory}/statepoint.100.h5")

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
fig.savefig(f'{args.directory}/XY_neutron_flux.png', dpi=300)

# Mesh neutron dose tally
dose_tally = sp.get_tally(name='mesh neutron dose tally')
dose = dose_tally.get_slice(scores=['flux']).mean.flatten()

dose_reshaped = dose.reshape(mesh.dimension[1], mesh.dimension[0])  # Note the order for plotting

# Convert to mrem/hr
dose_reshaped = dose_reshaped/mesh.volumes.reshape(mesh.dimension[1], mesh.dimension[0]) * 3600 / 10000000 # from pSv/s to mrem/hr
#log_dose = np.log10(dose_reshaped + 1e-20)  # Avoid log(0) issues

dose_log_norm = colors.LogNorm(vmin=0.1, vmax=10000)

dose_fig, dose_ax = plt.subplots(figsize=(8, 10))
dose_c = dose_ax.pcolormesh(x, y, dose_reshaped, shading='auto', cmap='magma', norm=dose_log_norm)
contours = dose_ax.contour(x[:-1], y[:-1], dose_reshaped, levels=[1, 10, 100, 1000, 10000], colors='white')
dose_ax.clabel(contours, inline=True, fontsize=18, inline_spacing=10)            
dose_ax.set_aspect('equal')
dose_fig.colorbar(dose_c, ax=dose_ax, label='Neutron Dose Rate (mrem/hr)')
dose_ax.set_xlabel('X (cm)')
dose_ax.set_ylabel('Y (cm)')
dose_ax.set_title('2D Neutron Dose Rate Distribution')
dose_fig.tight_layout()

dose_fig.savefig(f'{args.directory}/XY_neutron_dose_rate.png', dpi=300)

# Mesh photon dose tally
photon_dose_tally = sp.get_tally(name='mesh photon dose tally')
photon_dose = photon_dose_tally.get_slice(scores=['flux']).mean.flatten() 

photon_dose_reshaped = photon_dose.reshape(mesh.dimension[1], mesh.dimension[0])  # Note the order for plotting

photon_dose_reshaped = photon_dose_reshaped/mesh.volumes.reshape(mesh.dimension[1], mesh.dimension[0]) * 3600 / 10000000 # from pSv/s to mrem/hr

photon_dose_log_norm = colors.LogNorm(vmin=0.01, vmax=1000)

photon_dose_fig, photon_dose_ax = plt.subplots(figsize=(8, 10))
photon_dose_c = photon_dose_ax.pcolormesh(x, y, photon_dose_reshaped, shading='auto', cmap='magma', norm=photon_dose_log_norm)
photon_contours = photon_dose_ax.contour(x[:-1], y[:-1], photon_dose_reshaped, levels=[1, 10, 100, 1000, 10000], colors='white')
photon_dose_ax.clabel(photon_contours, inline=True, fontsize=18, inline_spacing=10)            
photon_dose_ax.set_aspect('equal')
photon_dose_fig.colorbar(photon_dose_c, ax=photon_dose_ax, label='Photon Dose Rate (mrem/hr)')
photon_dose_ax.set_xlabel('X (cm)')
photon_dose_ax.set_ylabel('Y (cm)')
photon_dose_ax.set_title('2D Photon Dose Rate Distribution')
photon_dose_fig.tight_layout()
photon_dose_fig.savefig(f'{args.directory}/XY_photon_dose_rate.png', dpi=300)

# Plot detector tally
det_volume = (np.pi * (1.5)**2) * 1  # cm^3

detector_tally = sp.get_tally(name='detector tally')
detector_flux_n = detector_tally.get_reshaped_data()[0, :, 0, 0, 0] 
detector_flux_p = detector_tally.get_reshaped_data()[0, :, 1, 0, 0]

detector_flux_n = detector_flux_n / det_volume 
detector_flux_p = detector_flux_p / det_volume  

det_spec_fig, det_spec_ax = plt.subplots(1, 2, figsize=(16,6))
det_spec_ax[0].spines['top'].set_visible(False)
det_spec_ax[0].spines['right'].set_visible(False)
det_spec_ax[1].spines['top'].set_visible(False)
det_spec_ax[1].spines['right'].set_visible(False)

energy_bins = detector_tally.find_filter(openmc.EnergyFilter).values
bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:]) / 1e6  # Convert from eV to MeV

det_spec_ax[0].step(bin_centers, detector_flux_n, where='mid')
det_spec_ax[0].set_xlim(0, 20)
det_spec_ax[0].set_yscale("log")
det_spec_ax[0].set_xlabel('Energy (MeV)')
det_spec_ax[0].set_ylabel('Neutron Flux (n/cm²-s)')
det_spec_ax[0].set_title('Neutron Energy Spectrum at Detector')

det_spec_ax[1].step(bin_centers, detector_flux_p, where='mid', color='orange')
det_spec_ax[1].set_yscale('log')
det_spec_ax[1].set_xscale('log')
det_spec_ax[1].set_xlabel('Energy (MeV)')
det_spec_ax[1].set_ylabel('Photon Flux (n/cm²-s)')
det_spec_ax[1].set_title('Photon Energy Spectrum at Detector')

det_spec_fig.tight_layout()
det_spec_fig.savefig(f'{args.directory}/detector_neutron_spectrum.png', dpi=300)




plt.show()

sp.close()


