from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
import random


parser = ArgumentParser(description='Convert dataset to KITTI format, optionnally create a visualization video',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_dir', metavar='DIR', type=Path, required=True,
                    help='folder containing the converted dataset')
parser.add_argument('--output_dir', metavar='DIR', type=Path, required=True,
                    help='foutput folder which will be used as a dataset for unsupervised depth training')
parser.add_argument('--scenes_list', type=Path, default=None,
                    help="List of folders containing videos to split, useful if some are used for testing")
parser.add_argument('--verbose', '-v', action='count', default=0)
parser.add_argument('--min_displacement', default=0, type=float,
                    help='Minimum displacement between two frames')
parser.add_argument('--max_rotation', default=1, type=float,
                    help='Maximum rotation between two frames. If above, frame is discarded and video is split')
parser.add_argument('--min_num_frames', default=5, type=int,
                    help='minimum number of frames in a video split. If below, split is removed')
parser.add_argument('--allow_interpolated_frames', action='store_true',
                    help='If set, will consider frames with interpolated odometry to be valid')
parser.add_argument('--adapt_min_disp_to_depth', action='store_true')
parser.add_argument('--train_split', type=float, default=0.9,
                    help='proportion of videos taken for training split, the other videos are taken for validation')
parser.add_argument('--seed', type=int, default=0)


def sample_splits(sequence, min_displacement, max_rot, min_num_frames):
    def get_rotation(row):
        flat_matrix = row[["pose00", "pose01", "pose02",
                           "pose10", "pose11", "pose12",
                           "pose20", "pose21", "pose22"]].values
        rotation = Rotation.from_matrix(flat_matrix.reshape(3, 3))
        return rotation
    current_split = [0]
    last_rot = None
    last_intrinsics = None
    for k in sample_frames(sequence, min_displacement):
        row = sequence.iloc[k]
        current_rot = get_rotation(row)
        current_intrinsics = row[["fx", "fy", "cx", "cy"]].values.astype(float)
        if last_rot is not None:
            rot_diff_mag = (current_rot.inv() * last_rot).magnitude()
        else:
            rot_diff_mag = 0
        if last_intrinsics is None:
            last_intrinsics = current_intrinsics

        if rot_diff_mag > max_rot or not np.allclose(current_intrinsics, last_intrinsics):
            if len(current_split) > min_num_frames:
                yield current_split
            current_split = []
            last_rot = None
            last_intrinsics = None
        else:
            last_rot = current_rot
            current_split.append(k)
    if len(current_split) > min_num_frames:
        yield current_split


def sample_frames(sequence, min_displacement):
    tvec = sequence[["pose03", "pose13", "pose23"]].values
    origin = tvec[0]
    current_disp = 0
    for j, pos in enumerate(tvec):
        current_disp = np.linalg.norm(pos - origin)
        if current_disp > min_displacement:
            yield j
            origin = pos


def get_min_depth(row):
    depth_path = row["image_path"].splitext() + ".npy"
    depth = np.load(depth_path)
    return depth.min()


def sample_frames_depth(sequence, min_displacement, min_depth):
    tvec = sequence[["pose03", "pose13", "pose23"]].values
    origin = tvec[0]
    current_disp = 0
    last_min_depth = get_min_depth(sequence.iloc[0])
    for i, pos in enumerate(tvec):
        current_min_depth = get_min_depth(sequence.iloc[i])
        if current_disp * np.min(last_min_depth, current_min_depth) / min_depth > min_displacement:
            yield i
            origin = pos
        current_disp = np.lialg.norm(pos - origin)


def main():
    args = parser.parse_args()
    random.seed(args.seed)
    args.output_dir.makedirs_p()
    total_scenes = 0
    total_frames = 0
    frames_per_video = {}
    dirs_per_video = {}
    if args.scenes_list is not None:
        with open(args.scenes_list, 'r') as f:
            dataset_folders = [args.dataset_dir / path[:-1] for path in f.readlines()]
    else:
        dataset_folders = [f for f in args.dataset_dir.walkdirs() if f.files("*.jpg")]
    for v in tqdm(dataset_folders):
        frames_per_video[v] = 0
        dirs_per_video[v] = []
        metadata = pd.read_csv(v / "metadata.csv")
        metadata["full_image_path"] = [v / Path(f).basename()
                                       for f in metadata["image_path"].values]
        valid_odometry_frames = metadata["registered"]
        if not args.allow_interpolated_frames:
            valid_odometry_frames = valid_odometry_frames & ~metadata["interpolated"]
        # Construct valid sequences
        valid_diff = valid_odometry_frames.astype(float).diff()
        invalidity_start = valid_diff.index[valid_diff == -1].tolist()
        validity_start = valid_diff.index[valid_diff == 1].tolist()
        if valid_odometry_frames.iloc[0]:
            validity_start = [0] + validity_start
        if valid_odometry_frames.iloc[-1]:
            invalidity_start.append(len(valid_odometry_frames))
        valid_sequences = [metadata.iloc[s:e].copy() for s, e in zip(validity_start, invalidity_start)]
        for i, seq in enumerate(valid_sequences):
            splits = sample_splits(seq, args.min_displacement, args.max_rotation, args.min_num_frames)
            for j, s in enumerate(splits):
                total_scenes += 1
                total_frames += len(s)
                frames_per_video[v] += len(s)
                final_seq = seq.iloc[s]
                relative_dir = args.dataset_dir.relpathto(v)
                folder_tree = relative_dir.splitall()[1:]
                folder_name = '_'.join(folder_tree) + '_{:02d}_{:02d}'.format(i, j)
                output_dir = args.output_dir / folder_name
                dirs_per_video[v].append(folder_name)
                output_dir.makedirs_p()
                for _, row in final_seq.iterrows():
                    image_name = Path(row["image_path"]).basename()
                    src_img_path = v / image_name
                    tgt_img_path = output_dir / image_name
                    src_depth_path = src_img_path.stripext() + '.npy'
                    tgt_depth_path = tgt_img_path.stripext() + '.npy'
                    src_img_path.copy(tgt_img_path)
                    src_depth_path.copy(tgt_depth_path)

                poses = final_seq[["pose00", "pose01", "pose02", "pose03",
                                   "pose10", "pose11", "pose12", "pose13",
                                   "pose20", "pose21", "pose22", "pose23"]]
                np.savetxt(output_dir / "poses.txt", poses)
                final_seq.to_csv(output_dir / "metadata.csv")
                intrinsics = final_seq[["fx", "fy", "cx", "cy"]].values[0]
                cam = np.array([[intrinsics[0], 0, intrinsics[2]],
                                [0, intrinsics[1], intrinsics[3]],
                                [0, 0, 1]])
                np.savetxt(output_dir / "cam.txt", cam)
    print("constructed a training set of {} frames in {} scenes".format(total_frames, total_scenes))

    random.shuffle(dataset_folders)
    train_frames = total_frames * args.train_split
    cumulative_num_frames = np.cumsum(np.array([frames_per_video[v] for v in dataset_folders]))
    used_for_train = (cumulative_num_frames - train_frames) < 0
    train_folders = [dirs_per_video[dataset_folders[i]] for i in np.where(used_for_train)[0]]
    val_folders = [dirs_per_video[dataset_folders[i]] for i in np.where(~used_for_train)[0]]
    with open(args.output_dir / "train.txt", 'w') as f:
        f.writelines([scene + '\n' for tf in train_folders for scene in tf])
    with open(args.output_dir / "val.txt", 'w') as f:
        f.writelines([scene + '\n' for vf in val_folders for scene in vf])


if __name__ == '__main__':
    main()
