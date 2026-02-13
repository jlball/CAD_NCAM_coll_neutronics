import openmc
import matplotlib.pyplot as plt
import numpy as np
import argparse
import copy
import os

parser = argparse.ArgumentParser(description='Run OpenMC simulation with DAGMC geometry.')

parser.add_argument("--ww", action="store_true", help="Use MAGIC method to geneerate weight windows for variance reduction")
parser.add_argument("--ww_method", default="magic", type=str, help="Weight window generation method (default: magic)")
parser.add_argument("--dagmc_file", "-f", default="BHDPE_dagmc.h5", help="Path to the DAGMC file (default: BHDPE_dagmc.h5)")
parser.add_argument("directory", help="Output file directory for OpenMC results")
parser.add_argument("--photons", action="store_true", help="Include photons in the simulation")
parser.add_argument("--batches", type=int, default=100, help="Number of batches for the OpenMC simulation (default: 100)")
parser.add_argument("--particles", type=int, default=100000, help="Number of particles per batch for the OpenMC simulation (default: 100000)")

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

B_HDPE = openmc.Material(material_id=7, name='B_HDPE')  # Keep original name for DAGMC mapping
B_HDPE.add_nuclide('C12', 0.321945534944)
B_HDPE.add_nuclide('C13', 0.003606465056)
B_HDPE.add_nuclide('H1', 0.62766123281334)
B_HDPE.add_nuclide('H2', 9.776718665999998e-05)
B_HDPE.add_nuclide('B10', 0.009253958)
B_HDPE.add_nuclide('B11', 0.037436042)
B_HDPE.set_density('g/cm3', 1)

materials_dict = {
    'aluminum': alumminum,
    'concrete': concrete,
    'carbon_fiber': carbon_fiber,
    'kretekast': kretekast,
    'deuterated_xylene': deuterated_xylene,
    "air": air,
    'B_HDPE': B_HDPE
}

def build_model(dagmc_file, source_position=(0, 120, 95), 
                source_strength=2e9, 
                simulate_photons=False, 
                ww=False, 
                ww_method="magic",
                batches=100,
                particles=100000):

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
    dagmc_univ = openmc.DAGMCUniverse(dagmc_file, auto_geom_ids=True)

    bounding_region = get_region_from_bbox(dagmc_univ.bounding_box, boundary_type='vacuum') & ~detector_region

    outer_cell = openmc.Cell(region=bounding_region)
    outer_cell.fill = dagmc_univ

    model.geometry = openmc.Geometry([outer_cell, detector_cell])

    # Automatically build Materials object to only include materials used in the simulation.
    DAGMC_mats = []
    for mat_name in dagmc_univ.material_names:
        if mat_name not in materials_dict:
            raise ValueError(f"Material '{mat_name}' from DAGMC file not found in materials dictionary. Please add it to the dictionary.")
        else:
            print(f"Mapping DAGMC material '{mat_name}' to OpenMC material '{materials_dict[mat_name].name}'")
            DAGMC_mats.append(materials_dict[mat_name])

    model.materials = openmc.Materials(DAGMC_mats + list(model.geometry.get_all_materials().values()))

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
    source.space = openmc.stats.Point(source_position)

    # source.angle = openmc.stats.PolarAzimuthal(
    #     mu=openmc.stats.Uniform(0, 1.0),  # Isotropic in the forward hemisphere
    #     phi=openmc.stats.Uniform(0, 2 * np.pi),
    #     reference_uvw=(0, -1, 0)
    # )

    source.angle = openmc.stats.Isotropic()  # Isotropic in all directions

    source.energy = openmc.stats.Discrete([14.06e6], [1.0])
    source.strength = source_strength # neutrons per second

    settings = openmc.Settings()
    settings.batches = batches
    settings.particles = particles
    settings.run_mode = 'fixed source'
    settings.source = source
    settings.photon_transport = simulate_photons

    model.settings = settings

    if ww:

        if ww_method.lower() == "fw_cadis":
            try:
                os.chdir(f"{args.directory}/ww_generation")
            except:
                os.makedirs(f"{args.directory}/ww_generation")
                os.chdir(f"{args.directory}/ww_generation")

            # Define weight window spatial mesh
            voxel_size = 25 # cm

            ww_mesh = openmc.RegularMesh()
            ww_mesh.dimension = (int((dagmc_univ.bounding_box.upper_right[0] - dagmc_univ.bounding_box.lower_left[0]) / voxel_size),
                                int((dagmc_univ.bounding_box.upper_right[1] - dagmc_univ.bounding_box.lower_left[1]) / voxel_size),
                                int((dagmc_univ.bounding_box.upper_right[2] - dagmc_univ.bounding_box.lower_left[2]) / voxel_size))
            ww_mesh.lower_left = dagmc_univ.bounding_box.lower_left
            ww_mesh.upper_right =  dagmc_univ.bounding_box.upper_right

            # Generate model for running random ray solve
            # Create a deep copy but fix material name mappings
            ww_model = copy.deepcopy(model)

            # Random ray solver requires a fixed source distribution, so we can use the same source but make it isotropic
            # ww_model.settings.source.angle = openmc.stats.Isotropic()

            print("##### CONVERT TO MGXS #####")
            ww_model.convert_to_multigroup(nparticles=100000,
                                        groups="CCFE-709")
            print("##### CONVERT TO RANDOM RAY #####")
            ww_model.convert_to_random_ray()
            ww_model.settings.random_ray["source_region_meshes"] = [(ww_mesh, [ww_model.geometry.root_universe])]
            #ww_model.settings.source = None  # No need for a source in the random ray solve, as it will be generated from the mesh

            # (Optional) Improve fidelity of the random ray solver by enabling linear sources
            ww_model.settings.random_ray['source_shape'] = 'linear'

            # (Optional) Increase the number of rays/batch, to reduce uncertainty
            ww_model.settings.particles = 100000

            # Create weight window object and adjust parameters
            wwg = openmc.WeightWindowGenerator(
                method='fw_cadis',
                mesh=ww_mesh,
                #max_realizations=settings.batches
            )

            # Add generator to Settings instance
            ww_model.settings.weight_window_generators = wwg
            print("##### RUNNING FW-CADIS WEIGHT WINDOW GENERATION #####")
            ww_model.run()

            os.chdir("../..")

            model.settings.weight_windows_file = f"{args.directory}/ww_generation/weight_windows.h5"
            model.settings.weight_windows_on = True
            model.settings.weight_window_checkpoints = {'collision': True, 'surface': True}
            model.settings.survival_biasing = False
        
        elif args.ww_method.lower() == "magic":
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

    # Tallies
    tallies = openmc.Tallies()

    # XY Mesh for flux and dose tallies
    mesh = openmc.RegularMesh()

    mesh.lower_left = (dagmc_univ.bounding_box.lower_left[0], dagmc_univ.bounding_box.lower_left[1], 0)
    mesh.upper_right = dagmc_univ.bounding_box.upper_right

    x_width =  mesh.upper_right[0] - mesh.lower_left[0] 
    y_width =  mesh.upper_right[1] - mesh.lower_left[1]

    mesh_size = 4 # cm

    mesh.dimension = (int(x_width/mesh_size), int(y_width/mesh_size), 1)  # Adjusted for better resolution

    mesh_filter = openmc.MeshFilter(mesh)

    energy_bins_n, dose_coeffs_n = openmc.data.dose_coefficients(
            particle="neutron", geometry="AP"
        )

    energy_bins_p, dose_coeffs_p = openmc.data.dose_coefficients(
            particle="photon", geometry="AP"
        )

    neutron_dose_filter = openmc.EnergyFunctionFilter(energy_bins_n, dose_coeffs_n)
    neutron_dose_filter.interpolation = "cubic"  # cubic interpolation is recommended by ICRP

    photon_dose_filter = openmc.EnergyFunctionFilter(energy_bins_p, dose_coeffs_p)
    photon_dose_filter.interpolation = "cubic"

    # Energy filter
    energy_filter = openmc.EnergyFilter.from_group_structure("CCFE-709") # 14.06 MeV neutrons

    # Particle filters
    neutron_filter = openmc.ParticleFilter('neutron')
    photon_filter = openmc.ParticleFilter('photon')
    dual_particle_filter = openmc.ParticleFilter(['neutron', 'photon'])

    # Mesh Flux
    mesh_flux_tally = openmc.Tally(name='mesh flux tally')
    mesh_flux_tally.filters = [mesh_filter, neutron_filter]
    mesh_flux_tally.scores = ['flux']  # Add dose-rate score

    # Mesh Neutron Dose
    mesh_neutron_dose = openmc.Tally(name='mesh neutron dose tally')
    mesh_neutron_dose.filters = [mesh_filter, neutron_dose_filter, neutron_filter]
    mesh_neutron_dose.scores = ['flux']  # Dose will be calculated via the

    mesh_photon_dose = openmc.Tally(name='mesh photon dose tally')
    mesh_photon_dose.filters = [mesh_filter, photon_dose_filter, photon_filter]
    mesh_photon_dose.scores = ['flux']  # Dose will be calculated via the dose coefficients

    # Detector Tally
    detector_tally = openmc.Tally(name='detector tally')
    detector_tally.filters = [openmc.CellFilter(detector_cell), energy_filter, dual_particle_filter]
    detector_tally.scores = ['flux']

    tallies.append(mesh_flux_tally)
    tallies.append(mesh_neutron_dose)
    tallies.append(mesh_photon_dose)
    tallies.append(detector_tally)
    model.tallies = tallies

    return model

if __name__ == "__main__":
    # Build OpenMC model object from DAGMC file and other settings
    model = build_model(args.dagmc_file, source_position=(0, 120, 95), source_strength=2e9, simulate_photons=args.photons, ww=args.ww, ww_method=args.ww_method)

    # Run OpenMC simulation
    model.run(cwd=args.directory)