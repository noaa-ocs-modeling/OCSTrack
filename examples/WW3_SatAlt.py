import numpy as np
from ocstrack.Model.model import WW3
from ocstrack.Observation.satellite import SatelliteData
from ocstrack.Collocation.collocate import Collocate

# 1. Define File Paths
#    Set the paths for your downloaded satellite data, model run, and where you want to save the collocated output.
sat_path = r"Your/Path/to/Downloaded Satellite Data/Here/"
model_path = r"Your/Path/to/WW3 run dir/Here/"
output_path =  r"Your/Path/Here/ww3_collocated.nc"
s_time,e_time = "2019-07-30", "2019-08-03"

# 2. Load Satellite Data
#    Initialize the SatelliteData object with your satellite data file.
sat_data = SatelliteData(sat_path)

# 3. Load Model Data
#    Instantiate the WW3 model object, specifying the run directory and model variable details.
model_run = WW3(
                    rundir=model_path,
                    model_dict={'var': 'hs'},
                    start_date=np.datetime64(s_time),
                    end_date=np.datetime64(e_time)
                  )

# 4. Perform Collocation
#    Create a Collocate object, providing the loaded model and satellite data.
coll = Collocate(
                 model_run=model_run,
                 observation=sat_data,
                 n_nearest=3,
                 temporal_interp=True
                 )
ds_coll = coll.run(output_path=output_path)
