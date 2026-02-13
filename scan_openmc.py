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
args = parser.parse_args()

source_x_positions = np.linspace(0, 3, num=6)  # 6 positions from 0 to 3 cm in x-direction

try:
    os.mkdir(args.directory)
except:
    print(f"Directory '{args.directory}' already exists. Simulation results will be stored there.")

for x_pos in source_x_positions:
    print(f"Running simulation with source at x = {x_pos} cm")
    model = build_model(args.dagmc_file, source_position=(x_pos, 120, 95), source_strength=2e9, simulate_photons=args.photons, ww=args.ww, ww_method=args.ww_method)
    model.run(cwd=f"{args.directory}/{x_pos}_cm_source_x")