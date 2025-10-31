import numpy as np
from ocstrack.Model.model import ADCSWAN
from ocstrack.Observation.satellite import SatelliteData
from ocstrack.Observation import get_sat
from ocstrack.Collocation.collocate import Collocate
# from ocstrack.utils import convert_longitude


# 1. Download Satellite Data
  #  Specify your desired date range, list of satellites, output directory, and geographical bounding box.
print("get_sat begin")
get_sat.get_multi_sat(start_date="2020-12-13",
                      end_date="2020-12-31",
                      sat_list=['sentinel3a','sentinel3b','jason2','jason3','cryosat2','saral','swot','sentinel6a'],
                      output_dir=r"Your/Path/where to save the sat altimetry/Here/",
                      lat_min=7,
                      lat_max=46,
                      lon_min=-98,
                      lon_max=-60,
                     )
print("get_sat end")

print("load sat begin")
# 2. Define File Paths
#    Set the paths for your downloaded satellite data, model run, and where you want to save the collocated output.
sat_path = r"Your/Path/to/Downloaded Satellite Data/Here/",
model_path = r"Your/Path/to/ADCIRC+SWAN run dir/Here/",
output_path =  r"Your/Path/Here/adcswn_collocated.nc",
s_time,e_time = "2019-07-30", "2019-08-03"

# 3. Load Satellite Data
#    Initialize the SatelliteData object with your satellite data file.
sat_data = SatelliteData(sat_path)
#    It's crucial to ensure longitude conventions match between satellite and model data.
#    Use convert_longitude if needed (mode=1 for converting to 0-360 degrees).
# sat_data.lon = convert_longitude(sat_data.lon, mode=1)

print("load model begin")
# 4. Load Model Data
#    Instantiate the SCHISM model object, specifying the run directory and model variable details.
model_run = ADCSWAN(
                    rundir=model_path,
                    model_dict={'var': 'swan_HS',
                                'startswith': 'swan_HS.63.nc',
                                # 'var_type': '2D',
                                'model': 'ADCSWAN'},
                    start_date=np.datetime64(s_time),
                    end_date=np.datetime64(e_time)
                  )

print("Collocation begin")
# 5. Perform Collocation
#    Create a Collocate object, providing the loaded model and satellite data.A 
# dist_coast = xr.open_dataset(r'Your/Path/to/OPTIONAL/distFromCoast.nc')
coll = Collocate(
                 model_run=model_run,
                 observation=sat_data,
                #  dist_coast=dist_coast,
                 n_nearest=3,
                 # search_radius = 3000,
                 temporal_interp=True
                 )
ds_coll = coll.run(output_path=output_path) # Execute the collocation and save the results