import openmc
import numpy as np
import argparse
import os
from run_openmc import build_model

parser = argparse.ArgumentParser(description="Run OpenMC simulations with varying source positions.")
parser.add_argument("directory", type=str, default="openmc_simulations", help="Directory to store OpenMC simulation results.")
parser.add_argument("--dagmc_file", "-f", type=str, required=True, help="Path to the DAGMC file.")
parser.add_argument("--photons", action="store_true", help="Enable photon transport in the simulation.")
parser.add_argument("--ww", action="store_true", help="Enable weight window generation.")
parser.add_argument("--ww_method", type=str, default="magic", choices=["magic", "fw_cadis"], help="Method for weight window generation (default: magic).")
parser.add_argument("--batches", type=int, default=100, help="Number of batches for the OpenMC simulation (default: 100).")
parser.add_argument("--particles", type=int, default=100000 , help="Number of particles per batch for the OpenMC simulation (default: 100000).")
args = parser.parse_args()

source_x_positions = np.linspace(0, 3, num=6)  # 6 positions from 0 to 3 cm in x-direction
sim_directories = [f"{args.directory}/{x_pos}_cm_source_x" for x_pos in source_x_positions]

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
        
        model = build_model(args.dagmc_file, 
                            source_position=(x_pos, 120, 95), 
                            source_strength=2e9, 
                            simulate_photons=args.photons, 
                            ww=args.ww, 
                            ww_method=args.ww_method,
                            batches=args.batches,
                            particles=args.particles)
        
        model.run(cwd=sim_directories[i])

for i, sim_dir in enumerate(sim_directories):
    print(f"Analyzing results in {sim_dir}")
    
    sp = openmc.StatePoint(f"{sim_dir}/statepoint.{args.batches}.h5")
