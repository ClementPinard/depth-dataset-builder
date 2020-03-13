from subprocess import PIPE, call
import pandas as pd
import numpy as np
from scipy import integrate
import tempfile


def extrapolate_position(speeds, timestamps, initial_position, final_position):
    trapz_x = integrate.cumtrapz(speeds[:, 0], timestamps, initial=0)
    trapz_y = integrate.cumtrapz(speeds[:, 1], timestamps, initial=0)
    trapz_z = integrate.cumtrapz(speeds[:, 2], timestamps, initial=0)
    trapz = np.stack([trapz_x, trapz_y, trapz_z], axis=-1)
    if initial_position is None and final_position is None:
        return trapz
    elif initial_position is not None and final_position is None:
        return trapz + initial_position
    elif initial_position is None and final_position is not None:
        return trapz + final_position - trapz[-1]
    else:
        interp = np.linspace(0, final_position - trapz[-1], speeds.shape[0])
        return trapz + interp


def preprocess_metadata(metadata, proj, centroid):
    def lambda_fun(x):
        return pd.Series(proj(*x), index=["x", "y"])
    position_xy = metadata[["location_longitude", "location_latitude"]].apply(lambda_fun, axis=1)
    metadata = metadata.join(position_xy)
    # Extrapolate position from speed and previous frames

    speed = metadata[["speed_east", "speed_north", "speed_down"]].values * np.array([1, 1, -1])
    timestamps = metadata["time"].values * 1e-6
    positions = metadata[["x", "y", "location_altitude"]].values
    if metadata["location_valid"].unique().tolist() == [0]:
        metadata["indoor"] = True
        positions = extrapolate_position(speed, timestamps, None, None)
    else:
        metadata["indoor"] = False
        if 0 in metadata["location_valid"].unique():
            location_validity = metadata["location_valid"].diff()

            invalidity_start = location_validity.index[location_validity == -1].tolist()
            validity_start = location_validity.index[location_validity == 1].tolist()

            if metadata["location_valid"].iloc[0] == 0:
                end = validity_start.pop(0)
                positions[:end] = extrapolate_position(speed[:end], timestamps[:end], None, positions[end-1])
            if metadata["location_valid"].iloc[-1] == 0:
                start = invalidity_start.pop(-1) - 1
                positions[start:] = extrapolate_position(speed[start:], timestamps[start:], positions[start], None)

            if(len(invalidity_start) != len(validity_start)):
                print("error")

            for start, end in zip(invalidity_start, validity_start):
                positions[start:end] = extrapolate_position(speed[start:end], timestamps[start:end], positions[start], positions[end-1])
        positions -= centroid
        print(positions)

    metadata["x"], metadata["y"], metadata["location_altitude"] = positions.transpose()

    return metadata


def extract_metadata(folder_path, file_path, native_wrapper, proj, w, h, f, centroid, save_path=None):
    metadata = native_wrapper.vmeta_extract(file_path)
    metadata = preprocess_metadata(metadata, proj, centroid)
    video_quality = h * w / f
    metadata["video_quality"] = video_quality
    metadata["height"] = h
    metadata["width"] = w
    metadata["framerate"] = f
    metadata["video"] = file_path
    metadata['frame'] = metadata.index + 1
    if save_path is not None:
        metadata.to_csv(save_path)
    return metadata
