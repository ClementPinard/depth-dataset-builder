from colmap_util import read_model as rm
import pandas as pd
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

parser = ArgumentParser(description='Filter COLMAP model of a single video by discards frames with impossible acceleration. '
                                    'The script then interplate dismissed frames and smooth out the trajectory with a SavGol filter.',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_images_colmap', metavar='FILE', type=Path, required=True,
                    help='Input COLMAP images.bin or images.txt file to filter.')
parser.add_argument('--metadata', metavar='FILE', type=Path, required=True,
                    help='Metadata CSV file of filtered video')
parser.add_argument('--output_images_colmap', metavar='FILE', type=Path, required=True,
                    help='Output images.bin or images.txt file with filtered frame localizations')
parser.add_argument('--interpolated_frames_list', type=Path, required=True,
                    help='Outpt list containing interpolated frames in order to discard them from ground-truth validation')
parser.add_argument('--filter_degree', default=3, type=int,
                    help='Degree of SavGol filter, higher means less filtering and more neighbouring frames')
parser.add_argument('--filter_time', default=0.1, type=float,
                    help='Time windows used by filter. Must be enough frames for filter degree')
parser.add_argument('--visualize', action="store_true")
parser.add_argument('--threshold_t', default=0.01, type=float,
                    help='Authorized deviation from SavGol filter output for position. Above, frame will be discarded')
parser.add_argument('--threshold_q', default=5e-3, type=float,
                    help='Same as threshold_t but for orientation with quaternions')


'''
def colmap_to_world(tvec, qvec):
    if any(np.isnan(qvec)):
        return qvec, tvec
    cam2world = rm.qvec2rotmat(qvec).T
    world_tvec = - cam2world @ tvec
    wolrd_qvec = np.array((qvec[0], *qvec[1:]))
    return wolrd_qvec, world_tvec
'''


def colmap_to_world(tvec, qvec):
    quats = Rotation.from_quat(qvec)
    return -quats.apply(tvec, inverse=True)


def world_to_colmap(tvec, qvec):
    quats = Rotation.from_quat(qvec)
    return -quats.apply(tvec, inverse=False)


def NEDtoworld(qvec):
    world2NED = np.float32([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])
    NED2world = world2NED
    world_qvec = rm.rotmat2qvec(NED2world @ rm.qvec2rotmat(qvec).T @ world2NED)
    return world_qvec


def quaternion_distances(prefix="", suffix=""):
    def fun(row):
        """ Create two Quaternions objects and calculate 3 distances between them """
        q1 = np.array((row[['{}q{}{}'.format(prefix, col, suffix) for col in list("wxyz")]]))
        q2 = np.array((row[['{}q{}2{}'.format(prefix, col, suffix) for col in list("wxyz")]]))
        row['{}qdist{}'.format(prefix, suffix)] = np.arccos(2 * (q1.dot(q2)) ** 2 - 1)

        return row
    return fun


def get_outliers(series, threshold):
    return ((series.diff(-1) > 0) & (series.diff(1) > 0) & (series > threshold)) | series.isnull()


def slerp_quats(quat_df, prefix=""):
    valid_df = quat_df[~quat_df["outlier"]]
    valid_index = valid_df.index
    total_index = quat_df.index

    # Note that scipy uses a different order convention than colmap : XYZW instead of WXYZ
    quats = Rotation.from_quat(valid_df[["{}q{}".format(prefix, col) for col in list("xyzw")]].values)
    slerp = Slerp(valid_index, quats)
    quats = slerp(total_index).as_quat()
    quats[quats[:, -1] < 0] *= -1
    return pd.DataFrame(slerp(total_index).as_quat(), index=quat_df.index)


def filter_colmap_model(input_images_colmap, output_images_colmap, metadata_path,
                        filter_degree=3, filter_time=0.1,
                        threshold_t=0.01, threshold_q=5e-3,
                        visualize=False, **env):
    if input_images_colmap.ext == ".txt":
        images_dict = rm.read_images_text(input_images_colmap)
    elif input_images_colmap.ext == ".bin":
        images_dict = rm.read_images_binary(input_images_colmap)
    else:
        print(input_images_colmap.ext)
    metadata = pd.read_csv(metadata_path).set_index("db_id", drop=False).sort_values("time")
    framerate = metadata["framerate"].iloc[0]
    filter_length = 2*int(filter_time * framerate) + 1

    image_df = pd.DataFrame.from_dict(images_dict, orient="index").set_index("id")
    image_df = image_df.reindex(metadata.index)
    metadata["outlier"] = image_df.isna().any(axis="columns")
    colmap_outliers = sum(metadata["outlier"])
    total_frames = len(metadata)
    image_df = image_df.dropna()
    tvec = np.stack(image_df["tvec"].values)
    qvec = np.stack(image_df["qvec"].values)

    # Check if quats are flipped by computing shifted dot product.
    # If no flip (and continuous orientation), the dot product is around 1. Otherwise, it's around -1
    # A quaternion is flipped if the dot product is negative

    flips = list((np.sum(qvec[1:] * qvec[:-1], axis=1) < 0).nonzero()[0] + 1)
    flips.append(qvec.shape[0])
    for j, k in zip(flips[::2], flips[1::2]):
        qvec[j:k] *= -1

    tvec_columns = ["colmap_tx", "colmap_ty", "colmap_tz"]
    quat_columns = ["colmap_qw", "colmap_qx", "colmap_qy", "colmap_qz"]
    metadata[tvec_columns] = pd.DataFrame(tvec, index=image_df.index)
    metadata[quat_columns] = pd.DataFrame(qvec, index=image_df.index)
    metadata["time (s)"] = metadata["time"] / 1e6
    metadata = metadata.set_index("time (s)")

    # Interpolate missing values for tvec and quat
    # In order to avoid extrapolation, we get rid of outlier at the beginning and the end of the sequence

    first_valid = metadata["outlier"].idxmin()
    last_valid = metadata["outlier"][::-1].idxmin()
    metadata = metadata.loc[first_valid:last_valid]

    metadata[tvec_columns] = metadata[tvec_columns].interpolate()
    metadata[["colmap_qx", "colmap_qy", "colmap_qz", "colmap_qw"]] = slerp_quats(metadata, prefix="colmap_")

    if visualize:
        metadata[["colmap_qw", "colmap_qx", "colmap_qy", "colmap_qz"]].plot()
        plt.gcf().canvas.set_window_title('Colmap quaternion (continuous)')

    qvec = metadata[["colmap_qx", "colmap_qy", "colmap_qz", "colmap_qw"]].values
    tvec = metadata[["colmap_tx", "colmap_ty", "colmap_tz"]].values

    world_tvec = colmap_to_world(tvec, qvec)

    world_tvec_filtered = savgol_filter(world_tvec, filter_length, filter_degree, axis=0)

    # TODO : this is linear filtering with renormalization,
    # mostly good enough but ideally should be something with slerp for quaternions
    qvec_filtered = savgol_filter(qvec, filter_length, filter_degree, axis=0)
    qvec_filtered = qvec_filtered / (np.linalg.norm(qvec_filtered, axis=-1, keepdims=True) + 1e-10)

    # Distances from raw and filtered values, we will dismiss those that are too far
    metadata["outlier_rot"] = np.arccos(2 * (np.sum(qvec * qvec_filtered, axis=1)) ** 2 - 1)

    metadata["outliers_pos"] = np.linalg.norm(world_tvec - world_tvec_filtered, axis=1)
    if visualize:
        metadata[["outliers_pos"]].plot()
        plt.gcf().canvas.set_window_title('difference between speed from colmap and from filtered')
        metadata[["outlier_rot"]].plot()
        plt.gcf().canvas.set_window_title('difference between rot speed from colmap and from filtered')

    metadata["tx"] = world_tvec[:, 0]
    metadata["ty"] = world_tvec[:, 1]
    metadata["tz"] = world_tvec[:, 2]
    metadata["qx"] = qvec[:, 0]
    metadata["qy"] = qvec[:, 1]
    metadata["qz"] = qvec[:, 2]
    metadata["qw"] = qvec[:, 3]

    if visualize:
        frame_q = metadata[["frame_quat_w", "frame_quat_x", "frame_quat_y", "frame_quat_z"]].values
        qref_list = []
        for q in frame_q:
            qref_list.append(NEDtoworld(q))
        qref = np.stack(qref_list)
        metadata["ref_qw"] = qref[:, 0]
        metadata["ref_qx"] = qref[:, 1]
        metadata["ref_qy"] = qref[:, 2]
        metadata["ref_qz"] = qref[:, 3]

        metadata["qw_filtered"] = qvec_filtered[:, 0]
        metadata["qx_filtered"] = qvec_filtered[:, 1]
        metadata["qy_filtered"] = qvec_filtered[:, 2]
        metadata["qz_filtered"] = qvec_filtered[:, 3]

        metadata["speed_up"] = -metadata["speed_down"]
        metadata[["speed_east", "speed_north", "speed_up"]].plot()
        plt.gcf().canvas.set_window_title('speed from sensors')
        colmap_speeds = framerate * metadata[["tx", "ty", "tz"]].diff()
        colmap_speeds.plot()
        plt.gcf().canvas.set_window_title('speeds from colmap (noisy)')
        metadata[["x", "y", "z", "tx", "ty", "tz"]].plot()
        plt.gcf().canvas.set_window_title('GPS(xyz) vs colmap position (tx,ty,tz)')
        metadata[["ref_qw", "ref_qx", "ref_qy", "ref_qz"]].plot()
        plt.gcf().canvas.set_window_title('quaternions from sensor')
        ax_q = metadata[["qw", "qx", "qy", "qz"]].plot()
        plt.gcf().canvas.set_window_title('quaternions from colmap vs from smoothed')

        metadata[["colmap_q{}2".format(col) for col in list('wxyz')]] = metadata[['colmap_qw',
                                                                                  'colmap_qx',
                                                                                  'colmap_qy',
                                                                                  'colmap_qz']].shift()
        metadata[["q{}2_filtered".format(col) for col in list('wxyz')]] = metadata[['qw_filtered',
                                                                                    'qx_filtered',
                                                                                    'qy_filtered',
                                                                                    'qz_filtered']].shift()
        metadata = metadata.apply(quaternion_distances(prefix="colmap_"), axis='columns')
        metadata = metadata.apply(quaternion_distances(suffix="_filtered"), axis='columns')
        ax_qdist = metadata[["colmap_qdist", "qdist_filtered"]].plot()
        plt.gcf().canvas.set_window_title('quaternions variations colmap vs filtered vs smoothed')

    metadata["outlier"] = metadata["outlier"] | \
        get_outliers(metadata["outliers_pos"], threshold_t) | \
        get_outliers(metadata["outlier_rot"], threshold_q)

    first_valid = metadata["outlier"].idxmin()
    last_valid = metadata["outlier"][::-1].idxmin()
    metadata = metadata.loc[first_valid:last_valid]

    metadata.loc[metadata["outlier"], ["tx", "ty", "tz", "qw", "qx", "qy", "qz"]] = np.NaN
    world_tvec_interp = metadata[["tx", "ty", "tz"]].interpolate(method="polynomial", order=3).values
    world_qvec_interp = slerp_quats(metadata).values

    world_tvec_smoothed = savgol_filter(world_tvec_interp, filter_length, filter_degree, axis=0)
    qvec_smoothed = savgol_filter(world_qvec_interp, filter_length, filter_degree, axis=0)
    qvec_smoothed /= np.linalg.norm(qvec_smoothed, axis=1, keepdims=True)

    colmap_tvec_smoothed = world_to_colmap(world_tvec_smoothed, qvec_smoothed)

    metadata["tx_smoothed"] = colmap_tvec_smoothed[:, 0]
    metadata["ty_smoothed"] = colmap_tvec_smoothed[:, 1]
    metadata["tz_smoothed"] = colmap_tvec_smoothed[:, 2]

    metadata["qx_smoothed"] = qvec_smoothed[:, 0]
    metadata["qy_smoothed"] = qvec_smoothed[:, 1]
    metadata["qz_smoothed"] = qvec_smoothed[:, 2]
    metadata["qw_smoothed"] = qvec_smoothed[:, 3]

    if visualize:
        metadata["world_tx_smoothed"] = world_tvec_smoothed[:, 0]
        metadata["world_ty_smoothed"] = world_tvec_smoothed[:, 1]
        metadata["world_tz_smoothed"] = world_tvec_smoothed[:, 2]
        metadata[["world_tx_smoothed", "world_ty_smoothed", "world_tz_smoothed"]].diff().plot()
        plt.gcf().canvas.set_window_title('speed from filtered and smoothed')
        metadata[["qw_smoothed", "qx_smoothed", "qy_smoothed", "qz_smoothed"]].plot(ax=ax_q)
        metadata[["q{}2_smoothed".format(col) for col in list('wxyz')]] = metadata[['qw_smoothed',
                                                                                    'qx_smoothed',
                                                                                    'qy_smoothed',
                                                                                    'qz_smoothed']].shift()
        metadata = metadata.apply(quaternion_distances(suffix="_smoothed"), axis='columns')
        metadata[["qdist_smoothed"]].plot(ax=ax_qdist)
        metadata[["outlier"]].astype(float).plot()
        plt.gcf().canvas.set_window_title('outliers indices')

    print("number of not localized by colmap : {}/{} ({:.2f}%)".format(colmap_outliers,
                                                                       total_frames,
                                                                       100 * colmap_outliers/total_frames))
    print("Total number of outliers : {} / {} ({:.2f}%)".format(sum(metadata["outlier"]),
                                                                total_frames,
                                                                100 * sum(metadata["outlier"])/total_frames))

    if visualize:
        plt.show()

    if output_images_colmap is not None:
        smoothed_images_dict = {}
        interpolated_frames = []
        for _, row in metadata.iterrows():
            db_id = row["db_id"]
            if row["outlier"]:
                interpolated_frames.append(row["image_path"])
            smoothed_images_dict[db_id] = rm.Image(id=db_id,
                                                   qvec=row[["qw_smoothed", "qx_smoothed", "qy_smoothed", "qz_smoothed"]].values,
                                                   tvec=row[["tx_smoothed", "ty_smoothed", "tz_smoothed"]].values,
                                                   camera_id=row["camera_id"],
                                                   name=row["image_path"],
                                                   xys=[], point3D_ids=[])
        if output_images_colmap.ext == ".txt":
            rm.write_images_text(smoothed_images_dict, output_images_colmap)
        elif output_images_colmap.ext == ".bin":
            rm.write_images_bin(smoothed_images_dict, output_images_colmap)
        else:
            print(output_images_colmap.ext)

    return interpolated_frames


if __name__ == '__main__':
    args = parser.parse_args()
    env = vars(args)
    interpolated_frames = filter_colmap_model(metadata_path=args.metadata, **env)
    with open(args.interpolated_frames_list, "w") as f:
        f.write("\n".join(interpolated_frames) + "\n")
