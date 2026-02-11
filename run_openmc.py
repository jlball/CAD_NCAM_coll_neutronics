import openmc
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run OpenMC simulation with DAGMC geometry.')

parser.add_argument("--ww", action="store_true", help="Use MAGIC method to geneerate weight windows for variance reduction")
parser.add_argument("--dagmc_file", "-f", default="dagmc.h5m", help="Path to the DAGMC file (default: dagmc.h5m)")
parser.add_argument("directory", help="Output file directory for OpenMC results")

args = parser.parse_args()


def get_region_from_bbox(bbox, boundary_type='vacuum'):
    xmin, ymin, zmin = bbox.lower_left
    xmax, ymax, zmax = bbox.upper_right

    x_min = openmc.XPlane(xmin, boundary_type=boundary_type)
    x_max = openmc.XPlane(xmax, boundary_type=boundary_type)
    y_min = openmc.YPlane(ymin, boundary_type=boundary_type)
    y_max = openmc.YPlane(ymax, boundary_type=boundary_type)
    z_min = openmc.ZPlane(zmin, boundary_type=boundary_type)
    z_max = openmc.ZPlane(zmax, boundary_type=boundary_type)

    bounding_box_region = +x_min & -x_max & +y_min & -y_max & +z_min & -z_max
    return bounding_box_region

# Materials
alumminum = openmc.Material(name='aluminum')
alumminum.add_element('Al', 1.0)
alumminum.set_density('g/cm3', 2.7)

# From PNNL Material Compendium (Ordinary Concrete)
concrete = openmc.Material(name='concrete')
concrete.add_element('H', 0.304245)
concrete.add_element('C', 0.002870)
concrete.add_element('O', 0.498628)
concrete.add_element('Na', 0.009179)
concrete.add_element("Mg", 0.000717)
concrete.add_element('Al', 0.010261)
concrete.add_element('Si', 0.150505)
concrete.add_element('K', 0.007114)
concrete.add_element('Ca', 0.014882)
concrete.add_element('Fe', 0.001599)

concrete.set_density('g/cm3', 2.3)

carbon_fiber = openmc.Material(name='carbon_fiber')
carbon_fiber.add_element('C', 0.9)
carbon_fiber.add_element('H', 0.1)
carbon_fiber.set_density('g/cm3', 1.6)

kretekast = concrete.clone()
kretekast.name = 'kretekast'

deuterated_xylene = openmc.Material(name='deuterated_xylene')
deuterated_xylene.add_element('C', 8)
deuterated_xylene.add_nuclide('H2', 10)
deuterated_xylene.set_density('g/cm3', 0.86)

air = openmc.Material(name='air')
air.add_element('N', 0.78)
air.add_element('O', 0.21)
air.add_element('Ar', 0.01)
air.set_density('g/cm3', 0.001225)

materials = openmc.Materials([alumminum, concrete, carbon_fiber, kretekast, deuterated_xylene, air])

model = openmc.model.Model()

# Detector region
det_thickness = 1 # cm
det_diameter = 3 #cm
detector_front = openmc.YPlane(-290)
detector_back = openmc.YPlane(-290 - det_thickness)
detector_cyl = openmc.YCylinder(r=det_diameter/2, z0=95)

detector_region = -detector_front & +detector_back & -detector_cyl
detector_cell = openmc.Cell(region=detector_region)
detector_cell.fill = deuterated_xylene

# Create OpenMC geometry from DAGMC file
dagmc_univ = openmc.DAGMCUniverse(args.dagmc_file, auto_geom_ids=True)

bounding_region = get_region_from_bbox(dagmc_univ.bounding_box, boundary_type='vacuum') & ~detector_region

outer_cell = openmc.Cell(region=bounding_region)
outer_cell.fill = dagmc_univ

model.geometry = openmc.Geometry([outer_cell, detector_cell])
model.materials = materials

# Plot the geometry
model.geometry.plot(
    #ax=ax,
    basis='yz',
    origin=(0, -150, 100),
    width=(700, 300),
    pixels=(2100, 1500),
    color_by='material',
)

plt.savefig('YZ_geometry.png', dpi=300)
plt.close()

# Settings
source = openmc.IndependentSource()
source.space = openmc.stats.Point((1, 120, 95))
source.angle = openmc.stats.PolarAzimuthal(
    mu=openmc.stats.Uniform(0, 1.0),  # Isotropic in the forward hemisphere
    phi=openmc.stats.Uniform(0, 2 * np.pi),
    reference_uvw=(0, -1, 0)
)
source.energy = openmc.stats.Discrete([14.06e6], [1.0])
source.strength = 2e9 # neutrons per second

settings = openmc.Settings()
settings.batches = 100
settings.particles = int(1e6)
settings.run_mode = 'fixed source'
settings.source = source

if args.ww:
    # Define weight window spatial mesh
    voxel_size = 50 # cm

    ww_mesh = openmc.RegularMesh()
    ww_mesh.dimension = (int((dagmc_univ.bounding_box.upper_right[0] - dagmc_univ.bounding_box.lower_left[0]) / voxel_size),
                        int((dagmc_univ.bounding_box.upper_right[1] - dagmc_univ.bounding_box.lower_left[1]) / voxel_size),
                        int((dagmc_univ.bounding_box.upper_right[2] - dagmc_univ.bounding_box.lower_left[2]) / voxel_size))
    ww_mesh.lower_left = dagmc_univ.bounding_box.lower_left
    ww_mesh.upper_right =  dagmc_univ.bounding_box.upper_right

    # Create weight window object and adjust parameters
    wwg = openmc.WeightWindowGenerator(
        method='magic',
        mesh=ww_mesh,
        max_realizations=settings.batches
    )

    # Add generator to Settings instance
    settings.weight_window_generators = wwg

model.settings = settings

# Tallies
tallies = openmc.Tallies()

# XY Mesh for flux and dose tallies
mesh = openmc.RegularMesh()

mesh.lower_left = (dagmc_univ.bounding_box.lower_left[0], dagmc_univ.bounding_box.lower_left[1], 0)
mesh.upper_right = dagmc_univ.bounding_box.upper_right

x_width =  mesh.upper_right[0] - mesh.lower_left[0] 
y_width =  mesh.upper_right[1] - mesh.lower_left[1]

mesh_size = 2 # cm

mesh.dimension = (int(x_width/mesh_size), int(y_width/mesh_size), 1)  # Adjusted for better resolution

mesh_filter = openmc.MeshFilter(mesh)

energy_bins_n, dose_coeffs_n = openmc.data.dose_coefficients(
        particle="neutron", geometry="AP"
    )

neutron_dose_filter = openmc.EnergyFunctionFilter(energy_bins_n, dose_coeffs_n)
neutron_dose_filter.interpolation = "cubic"  # cubic interpolation is recommended by ICRP

# Energy filter
energy_filter = openmc.EnergyFilter.from_group_structure("CCFE-709") # 14.06 MeV neutrons

# Mesh Flux
mesh_flux_tally = openmc.Tally(name='mesh flux tally')
mesh_flux_tally.filters = [mesh_filter]
mesh_flux_tally.scores = ['flux']  # Add dose-rate score

# Mesh Neutron Dose
mesh_neutron_dose = openmc.Tally(name='mesh neutron dose tally')
mesh_neutron_dose.filters = [mesh_filter, neutron_dose_filter]
mesh_neutron_dose.scores = ['flux']  # Dose will be calculated via the

# Detector Tally
detector_tally = openmc.Tally(name='detector tally')
detector_tally.filters = [openmc.CellFilter(detector_cell), energy_filter]
detector_tally.scores = ['flux']

tallies.append(mesh_flux_tally)
tallies.append(mesh_neutron_dose)
tallies.append(detector_tally)
model.tallies = tallies

# Run OpenMC simulation
model.run(cwd=args.directory)