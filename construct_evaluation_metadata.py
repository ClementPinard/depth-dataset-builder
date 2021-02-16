from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import random
import pandas as pd
from tqdm import tqdm
import numpy as np


parser = ArgumentParser(description='Create a file list for tests based on a set of movement constraints, create a FPV file',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_dir', metavar='DIR', type=Path, required=True,
                    help='folder containing the converted dataset')
parser.add_argument('--split', type=float, default=0,
                    help="proportion between train and test. By default, the whole dataset serves for evaluation")
parser.add_argument('--seed', type=int, default=0,
                    help='seed for random classification between train and val')
parser.add_argument('--verbose', '-v', action='count', default=0)
parser.add_argument('--max_num_samples', default=500, type=int)
parser.add_argument('--min_shift', default=0, type=int,
                    help='Minimum of former frames with valid odometry')
parser.add_argument('--allow_interpolated_frames', action='store_true',
                    help='If set, will consider frames with interpolated odometry to be valid')


def flight_path_vector(sequence, max_shift):
    """Get the Flight Path Vector for each frame of the sequence

    Args:
        sequence (pd.DataFrame): table of a particular sequence, with pose and intrinsics info
        max_shift (int): max shift that will be used to deduce the mean speed vector of the camera

    Returns:
        [type]: [description]
    """
    tvec = sequence[["pose03", "pose13", "pose23"]]
    displacement = np.zeros_like(tvec)
    for j in range(1, max_shift):
        displacement += tvec.diff(j) / j
    # TODO Note that this is only valid for pinhole cameras.
    fpv_x = sequence["fx"] * displacement["pose03"] / displacement["pose23"] + sequence["cx"]
    fpv_y = sequence["fy"] * displacement["pose13"] / displacement["pose23"] + sequence["cy"]
    return fpv_x, fpv_y


def main():
    args = parser.parse_args()
    random.seed(args.seed)
    folder_tree = args.dataset_dir.walkdirs()
    video_sequences = []
    for f in folder_tree:
        if f.files('*.jpg'):
            video_sequences.append(f)

    # Select a subset of the videos for training. It won't be used for 
    # constructing the validation set.
    random.shuffle(video_sequences)
    n = len(video_sequences)
    train_videos = video_sequences[:int(n*args.split)]
    test_videos = video_sequences[int(n*args.split):]

    total_valid_frames = []
    for v in tqdm(test_videos):
        metadata = pd.read_csv(v / "metadata.csv")
        # Construct table of frames with valid odometry
        # If option selected, this includes the interpolated frames
        valid_odometry_frames = metadata["registered"]
        if not args.allow_interpolated_frames:
            valid_odometry_frames = valid_odometry_frames & ~metadata["interpolated"]
        valid_diff = valid_odometry_frames.astype(float).diff()
        # Get start and end of validity.
        # The way we can reconstruct sequences only containing valid frames
        invalidity_start = valid_diff.index[valid_diff == -1].tolist()
        validity_start = valid_diff.index[valid_diff == 1].tolist()
        if valid_odometry_frames.iloc[0]:
            validity_start = [0] + validity_start
        if valid_odometry_frames.iloc[-1]:
            invalidity_start.append(len(valid_odometry_frames))
        # valid_sequences is a list of dataframes with only frames with valid odometry
        valid_sequences = [metadata.iloc[s:e].copy() for s, e in zip(validity_start, invalidity_start)]
        for s in valid_sequences:
            fpv_x, fpv_y = flight_path_vector(s, max_shift=3)
            s["fpv_x"] = fpv_x
            s["fpv_y"] = fpv_y

            # Get valid frames for depth :
            # - Has more than <min_shift> frames before it that have a valid odometry
            # This is useful for algorithms that require multiple frames with known odometry.
            # For that we just discard the first <min_shift> frames of the sequence.
            # - Is not interpolated
            valid_frames = s.iloc[args.min_shift:]
            valid_frames = valid_frames[~valid_frames["interpolated"]]
            valid_frames["image_path"] = [args.dataset_dir.relpathto(v) / Path(f).basename()
                                          for f in valid_frames["image_path"].values]

            # Add the valid frames of this sequence to the list of all valid frames
            total_valid_frames.append(valid_frames)

    total_valid_frames_df = pd.concat(total_valid_frames)

    # Select a subset of this tables for evaluation. Each row represent a frame that we can use for depth
    # evaluation if we want.
    if len(total_valid_frames_df) <= args.max_num_samples:
        # We don't have enough valid frames, just take all the frames
        print("Warning : Dataset has not enough valid frames, "
              "constructing a test set with only {} frames".format(len(total_valid_frames_df)))
        final_frames = total_valid_frames_df
    else:
        final_frames = total_valid_frames_df.sample(args.max_num_samples, random_state=args.seed)

    # Construct final metdata :
    train_dirs_list_path = args.dataset_dir / "train_folders.txt"
    image_list_path = args.dataset_dir / "test_files.txt"
    fpv_list_path = args.dataset_dir / "fpv.txt"
    with open(image_list_path, 'w') as f:
        f.writelines(line + "\n" for line in final_frames["image_path"].values)
    np.savetxt(fpv_list_path, final_frames[["fpv_x", "fpv_y"]].values)
    if len(train_videos) > 0:
        with open(train_dirs_list_path, 'w') as f:
            f.writelines([folder + "\n" for folder in train_videos])


if __name__ == '__main__':
    main()
