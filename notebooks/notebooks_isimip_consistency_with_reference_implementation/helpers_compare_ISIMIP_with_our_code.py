from cf_units import num2date
import iris
import numpy as np

def get_dates(x):
    time_dimension = x.coords()[2]
    dates = time_dimension.units.num2date(time_dimension.points)
    return dates

get_dates = np.vectorize(get_dates)


def read_in_and_preprocess_isimip_testing_data_with_dates(variable, isimip_data_path = "isimip3basd-master/data/"):
    
    obs = iris.load_cube(isimip_data_path+variable+"_obs-hist_coarse_1979-2014.nc")
    cm_hist = iris.load_cube(isimip_data_path+variable+"_sim-hist_coarse_1979-2014.nc")
    cm_future = iris.load_cube(isimip_data_path+variable+"_sim-fut_coarse_2065-2100.nc")

    dates = {
        "time_obs_hist": get_dates(obs),
        "time_cm_hist": get_dates(cm_hist),
        "time_cm_future": get_dates(cm_future)
    }
    
    obs = np.array(obs.data)
    cm_hist = np.array(cm_hist.data)
    cm_future = np.array(cm_future.data)

    
    obs = np.moveaxis(obs, -1, 0)
    cm_hist = np.moveaxis(cm_hist, -1, 0)
    cm_future = np.moveaxis(cm_future, -1, 0)
    
    return obs, cm_hist, cm_future, dates


def read_in_debiased_testing_data_with_dates(variable, isimip_data_path = "isimip3basd-master/data/debiased_models/"):
    
    debiased_data = iris.load_cube(isimip_data_path+variable+"_sim-fut-basd_coarse_2065-2100.nc")

    dates = get_dates(debiased_data)
    debiased_data = np.array(debiased_data.data)
    debiased_data = np.moveaxis(debiased_data, -1, 0)
    
    return debiased_data, dates