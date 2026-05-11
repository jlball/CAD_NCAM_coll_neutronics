from cad_to_dagmc import CadToDagmc
import openmc
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Convert CAD to DAGMC and visualize with OpenMC")
parser.add_argument("--step-file", type=str, default="NCAM_collimator_test.step",
                    help="Path to the STEP file to be converted")
parser.add_argument("--output-file", type=str, default="dagmc.h5m",
                    help="Output DAGMC H5M file name")
args = parser.parse_args()

# Load STEP file and mesh for DAGMC
model = CadToDagmc()
model.add_stp_file(
    filename=args.step_file,
    material_tags=["concrete", "aluminum", "aluminum", "carbon_fiber", "aluminum", "kretekast", "stainless_steel"]
)

# material_tags=["concrete", "aluminum", "aluminum", "carbon_fiber", "aluminum", "kretekast"]
# material_tags=["concrete", "air", "aluminum", "carbon_fiber", "air", "kretekast"]

model.export_dagmc_h5m_file(
    filename=args.output_file,
    meshing_backend="gmsh",  # Default
    min_mesh_size=0.1,
    max_mesh_size=5.0,
    scale_factor=0.1,
    set_size={
        2 : 0.2,
        3 : 2.0,
        4 : 0.5,
        5 : 0.5,
        6 : 0.2,
    }
)
