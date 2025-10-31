import numpy as np
from ocstrack.Model.model import SCHISM
from ocstrack.Observation.argofloat import ArgoData
from ocstrack.Observation import get_argo
from ocstrack.Collocation.collocate import Collocate
from ocstrack.utils import convert_longitude


# Download Argo Data
# Specify your desired date range, region, output directory, and geographical bounding box.
print("get_argo begin")
get_argo.get_argo(start_date="2019-08-29",
                      end_date="2019-10-05",
                      region = "pacific_ocean",
                      output_dir=r"Your/Path/where to save the sat altimetry/Here/",
                      lat_min=50,
                      lat_max=59,
                      lon_min=165,
                      lon_max=-165,
                     )
print("get_argo end")

print("load argo begin")
# Define File Paths
argo_path = r"Your/Path/to/Downloaded/pacific_ocean/processed/"
model_path = r"Your/Path/to/SCHISM run dir/Here/",
output_path =  r"Your/Path/Here/schism_collocated_argo.nc",
s_time,e_time = "2019-09-15", "2019-09-30"
print("load data begin")

print("load argo begin")
argo_data = ArgoData(argo_path)
print("Argo data loaded.")
#    Use convert_longitude if needed (mode=1 for converting to 0-360 degrees).
argo_data.lon = convert_longitude(argo_data.lon, mode=1)

print("load model begin")
model_config_3d = {
    'var': 'temperature',
    'startswith': 'temperature_',
    'var_type': '3D_Profile',
    'zcor_var': 'zCoordinates',
    'zcor_startswith': 'zCoordinates_'
}
model_run = SCHISM(
    rundir=model_path,
    model_dict=model_config_3d,
    start_date=np.datetime64(s_time),
    end_date=np.datetime64(e_time)
)

# print("Converting model mesh longitudes...")
# model_run.mesh_x = convert_longitude(model_run.mesh_x, mode=2) 
# print("Model loaded.")

print(f"Argo LONS --- min: {argo_data.lon.min()}, max: {argo_data.lon.max()}")
print(f"Model LONS --- min: {model_run.mesh_x.min()}, max: {model_run.mesh_x.max()}")

print("Initializing Collocate class...")
# dist_coast = xr.open_dataset(r'Your/Path/to/OPTIONAL/distFromCoast.nc')
try:
    collocator = Collocate(
        model_run=model_run,
        observation=argo_data,
        # dist_coast=dist_coast,
        n_nearest=3,
        temporal_interp=True
    )
    print("Collocate class initialized.")
except Exception as e:
    print(f"Error during Collocator initialization: {e}")
print("Starting collocation run...")

try:
    collocated_data = collocator.run(output_path=output_path)
    print("Collocation run complete.")
    print(f"Collocation finished. Output saved to: {output_path}")
    print("\n--- Collocated Dataset ---")
    print(collocated_data)
except Exception as e:
    print(f"Error during collocation run: {e}")