import openmc
import numpy as np
import argparse
import os
from run_openmc import build_model
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser(description="Run OpenMC simulations with varying source positions.")
parser.add_argument("directory", type=str, default="openmc_simulations", help="Directory to store OpenMC simulation results.")
parser.add_argument("--dagmc_file", "-f", type=str, help="Path to the DAGMC file.")
parser.add_argument("--photons", action="store_true", help="Enable photon transport in the simulation.")
parser.add_argument("--ww", action="store_true", default=False, help="Enable weight window generation.")
parser.add_argument("--ww_method", type=str, default="magic", choices=["magic", "fw_cadis", "pre-generated"], help="Method for weight window generation (default: magic).")
parser.add_argument("--ww_path", type=str, help="Path to pre-generated weight window file (required if ww_method is 'pre-generated').")
parser.add_argument("--batches", type=int, default=100, help="Number of batches for the OpenMC simulation (default: 100).")
parser.add_argument("--particles", type=int, default=100000 , help="Number of particles per batch for the OpenMC simulation (default: 100000).")
parser.add_argument("--postprocess", action="store_true", help="Run post-processing and plotting after simulations.")
parser.add_argument("--low_energy", type=float, default=None, help="Lower energy bound for neutron transport")
args = parser.parse_args()

#source_x_positions = [0.00, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # cm

source_x_positions = [0.00, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # 11 positions from 0 to 1 cm

sim_directories = [f"{args.directory}/{x_pos}_cm_source_x" for x_pos in source_x_positions]

if not args.postprocess:
    run_simulation = True 

    try:
        os.mkdir(args.directory)
    except:
        overwrite = input(f"Directory '{args.directory}' already exists. Do you want to run simulations and overwrite existing results? (y/n) ")
        if overwrite.lower() != 'y':
            run_simulation = False

    if run_simulation:
        for i, x_pos in enumerate(source_x_positions):
            print(f"Running simulation with source at x = {x_pos} cm")

            if args.ww and not args.ww_method.lower() == "pre-generated":
                if i == 0:
                    print("Generating weight windows for the first simulation...")
                    model = build_model(args.dagmc_file, 
                                        source_position=(x_pos, 150, 95), 
                                        source_strength=2e9, 
                                        simulate_photons=args.photons, 
                                        ww=args.ww, 
                                        ww_method=args.ww_method,
                                        batches=args.batches,
                                        particles=args.particles,
                                        low_energy=args.low_energy)

                else:
                    print("Reusing weight windows from previous simulation...")
                    model = build_model(args.dagmc_file, 
                                        source_position=(x_pos, 150, 95), 
                                        source_strength=2e9, 
                                        simulate_photons=args.photons, 
                                        ww=args.ww, 
                                        ww_method="pre-generated",
                                        ww_path=f"{sim_directories[0]}/weight_windows.h5",
                                        batches=args.batches,
                                        particles=args.particles,
                                        low_energy=args.low_energy)

            if args.ww and args.ww_method.lower() == "pre-generated":
                print("Using user-provided pre-generated weight windows...")
                model = build_model(args.dagmc_file, 
                                    source_position=(x_pos, 150, 95), 
                                    source_strength=2e9, 
                                    simulate_photons=args.photons, 
                                    ww=args.ww, 
                                    ww_method="pre-generated",
                                    ww_path=args.ww_path,
                                    batches=args.batches,
                                    particles=args.particles,
                                    low_energy=args.low_energy)
                
            else:
                model = build_model(args.dagmc_file, 
                                    source_position=(x_pos, 150, 95), 
                                    source_strength=2e9, 
                                    simulate_photons=args.photons, 
                                    ww=args.ww,
                                    ww_method=args.ww_method,
                                    batches=args.batches,
                                    particles=args.particles,
                                    low_energy=args.low_energy)

            model.run(cwd=sim_directories[i])


# Setup figures for plotting results
det_spec_fig, det_spec_ax = plt.subplots(figsize=(8, 6))
det_spec_ax.spines['top'].set_visible(False)
det_spec_ax.spines['right'].set_visible(False)
det_spec_ax.set_xlabel('Energy (MeV)')
det_spec_ax.set_ylabel('Neutron Flux (n/cm²-s)')
det_spec_ax.set_title('Neutron Energy Spectrum at Detector')
#det_spec_ax.set_xscale('log')
det_spec_ax.set_xlim(2, 18)
det_spec_ax.set_yscale('log')


direct_flux = []
direct_flux_std_dev = []

E_threshold = 2.5  # MeV

for i, sim_dir in enumerate(sim_directories):
    print(f"Analyzing results in {sim_dir}")
    
    sp = openmc.StatePoint(f"{sim_dir}/statepoint.{args.batches}.h5")

    # Analyze detector tally
    det_volume = (np.pi * (1.5)**2) * 1  # cm^3
    detector_tally = sp.get_tally(name='detector tally')
    detector_flux_n = detector_tally.get_reshaped_data()[0, :, 0, 0, 0] 
    detector_flux_n_std_dev = detector_tally.get_reshaped_data(value="std_dev")[0, :, 0, 0, 0]

   # detector_flux_p = detector_tally.get_reshaped_data()[0, :, 1, 0, 0]

    detector_flux_n = detector_flux_n / det_volume
    detector_flux_n_std_dev = detector_flux_n_std_dev / det_volume 
    # detector_flux_p = detector_flux_p / det_volume  

    energy_bins = detector_tally.find_filter(openmc.EnergyFilter).values
    bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:]) / 1e6  # Convert from eV to MeV

    # Plot neutron and photon spectra
    det_spec_ax.step(bin_centers, detector_flux_n, label=f'Source at x={source_x_positions[i]:.2f} cm')
    det_spec_ax.fill_between(bin_centers, detector_flux_n - detector_flux_n_std_dev, detector_flux_n + detector_flux_n_std_dev, alpha=0.3, step='pre')

    direct_flux.append(np.sum(detector_flux_n[bin_centers > E_threshold]))  
    direct_flux_std_dev.append(np.sqrt(np.sum(np.power(detector_flux_n_std_dev[bin_centers > E_threshold], 2))))

det_spec_ax.set_ybound(lower = 0.0001)
det_spec_ax.legend()
det_spec_fig.savefig(f"{args.directory}/detector_neutron_spectrum.png", dpi=300)

fig, ax = plt.subplots(figsize=(8, 6))
ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)

ax.scatter(source_x_positions, direct_flux, marker='o', color='blue')
ax.vlines(1.06, ymin=0, ymax=max(direct_flux)*1.1, colors='red', linestyles='dashed', label='FOV Edge (x=1.06 cm)')
ax.set_xlabel('Source X Position (cm)', fontsize=14)
ax.set_ylabel('Direct Neutron Flux at Detector (n/cm²-s)', fontsize=14)
#ax.set_yscale('log')
#ax.set_title('Direct Neutron Flux at Detector vs Source X Position')
ax.set_ybound(lower=0)
fig.savefig(f"{args.directory}/direct_neutron_flux_vs_source_position.png", dpi=300)

data_dict = {
    "source_x_positions": source_x_positions,
    "direct_flux": direct_flux,
    "direct_flux_std_dev": direct_flux_std_dev
}

with open(f"{args.directory}/direct_flux_data.pkl", "wb") as f:
    pickle.dump(data_dict, f)