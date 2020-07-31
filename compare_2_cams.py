from colmap_util import read_model as rm
import pandas as pd
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

parser = ArgumentParser(description='Take all the drone videos of a folder and put the frame '
                                    'location in a COLMAP file for vizualisation',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_images_1', metavar='FILE', type=Path)
parser.add_argument('--input_images_2', metavar='FILE', type=Path)
parser.add_argument('--metadata_path_1', metavar='FILE', type=Path)
parser.add_argument('--metadata_path_2', metavar='FILE', type=Path)
parser.add_argument('--visualize', action="store_true")
parser.add_argument('--interpolated_frames_list', type=Path)


def colmap_to_world(tvec, qvec):
    quats = Rotation.from_quat(qvec)
    return -quats.apply(tvec, inverse=True)


def qvec_diff(qvec1, qvec2):
    rot1 = Rotation.from_quat(qvec1)
    rot2 = Rotation.from_quat(qvec2)

    diff = rot1 * rot2.inv()

    return diff.as_quat()


def open_video(images_colmap, metadata_path):
    if images_colmap.ext == ".txt":
        images_dict = rm.read_images_text(images_colmap)
    elif images_colmap.ext == ".bin":
        images_dict = rm.read_images_binary(images_colmap)
    else:
        print(images_colmap.ext)
    image_df = pd.DataFrame.from_dict(images_dict, orient="index").set_index("id")
    metadata = pd.read_csv(metadata_path).set_index("db_id", drop=False).sort_values("time")

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

    qvec = metadata[["colmap_qx", "colmap_qy", "colmap_qz", "colmap_qw"]].values
    tvec = metadata[["colmap_tx", "colmap_ty", "colmap_tz"]].values

    world_tvec = colmap_to_world(tvec, qvec)

    metadata["tx"] = world_tvec[:, 0]
    metadata["ty"] = world_tvec[:, 1]
    metadata["tz"] = world_tvec[:, 2]
    metadata["qx"] = qvec[:, 0]
    metadata["qy"] = qvec[:, 1]
    metadata["qz"] = qvec[:, 2]
    metadata["qw"] = qvec[:, 3]
    return metadata


def compare_metadata(metadata_1, metadata_2):
    qvec_1 = metadata_1[["qx", "qy", "qz", "qw"]].values
    qvec_2 = metadata_2[["qx", "qy", "qz", "qw"]].values

    qvec = pd.DataFrame(qvec_diff(qvec_1, qvec_2), index=metadata_1.index)
    tvec = metadata_1[["tx", "ty", "tz"]] - metadata_2[["tx", "ty", "tz"]]

    return qvec, tvec


def compare_cams(input_images_1, input_images_2, metadata_path_1, metadata_path_2,
                 **env):

    metadata_1 = open_video(input_images_1, metadata_path_1)
    metadata_2 = open_video(input_images_2, metadata_path_2)
    assert(len(metadata_1) == len(metadata_2))
    metadata_1[["qw", "qx", "qy", "qz"]].plot()
    metadata_2[["qw", "qx", "qy", "qz"]].plot()
    metadata_1[["tx", "ty", "tz"]].plot()
    metadata_2[["tx", "ty", "tz"]].plot()

    qvec, tvec = compare_metadata(metadata_1, metadata_2)
    qvec.plot()
    tvec.plot()
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    env = vars(args)
    compare_cams(**env)
