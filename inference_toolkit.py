import numpy as np
from path import Path
from imageio import imread
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class Timer:
    def __init__(self):
        self._start_time = None
        self._elapsed_time = 0

    def running(self):
        return self._start_time is not None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            return

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            return

        self._elapsed_time += time.perf_counter() - self._start_time
        self._start_time = None

    def get_elapsed(self):
        return self._elapsed_time

    def reset(self):
        self.__init__()


class inferenceFramework(object):
    def __init__(self, root, test_files, min_depth=1e-3, max_depth=80, max_shift=50, frame_transform=None):
        self.root = Path(root)
        self.test_files = test_files
        self.min_depth, self.max_depth = min_depth, max_depth
        self.max_shift = max_shift
        self.frame_transform = frame_transform

    def __getitem__(self, i):
        timer = Timer()
        sample = inferenceSample(self.root, self.test_files[i], self.max_shift, timer, self.frame_transform)
        sample.timer.start()
        return sample

    def finish_frame(self, sample):
        sample.timer.stop()
        return sample.timer.get_elapsed()

    def __len__(self):
        return len(self.img_files)


class inferenceSample(object):
    def __init__(self, root, file, max_shift, timer, frame_transform=None):
        self.root = root
        self.file = file
        self.frame_transform = frame_transform
        self.timer = timer
        full_filepath = self.root / file
        scene = full_filepath.parent
        scene_files = sorted(scene.files("*.jpg"))
        poses = np.genfromtxt(scene / "poses.txt").reshape((-1, 3, 4))
        sample_id = scene_files.index(full_filepath)
        assert(sample_id > max_shift)
        start_id = sample_id - max_shift
        self.valid_frames = scene_files[start_id:sample_id + 1][::-1]
        valid_poses = np.flipud(poses[start_id:sample_id + 1])
        last_line = np.broadcast_to(np.array([0, 0, 0, 1]), (valid_poses.shape[0], 1, 4))
        valid_poses_full = np.concatenate([valid_poses, last_line], axis=1)
        self.poses = (np.linalg.inv(valid_poses_full[0]) @  valid_poses_full)[:, :3]
        R = self.poses[:, :3, :3]
        self.rotation_angles = Rotation.from_matrix(R).magnitude()
        self.displacements = np.linalg.norm(self.poses[:, :, -1], axis=-1)

        if (scene / "intrinsics.txt").isfile():
            self.intrinsics = np.stack([np.genfromtxt(scene / "intrinsics.txt")]*max_shift)
        else:
            intrinsics_files = [f.stripext() + "_intrinsics.txt" for f in self.valid_frames]
            self.intrinsics = np.stack([np.genfromtxt(i) for i in intrinsics_files])

    def timer_decorator(func, *args, **kwargs):
        def wrapper(self, *args, **kwargs):
            if self.timer.running():
                self.timer.stop()
                res = func(self, *args, **kwargs)
                self.timer.start()
            else:
                res = func(self, *args, **kwargs)
            return res
        return wrapper

    @timer_decorator
    def get_frame(self, shift=0):
        file = self.valid_frames[shift]
        img = imread(file)
        if self.frame_transform is not None:
            img = self.frame_transform(img)
        return img, self.intrinsics[shift], self.poses[shift]

    @timer_decorator
    def get_previous_frame(self, shift=1, displacement=None, max_rot=1):
        if displacement is not None:
            shift = max(1, np.abs(self.displacements - displacement).argmin())
        rot_valid = self.rotation_angles < max_rot
        assert sum(rot_valid[1:shift+1] > 0), "Rotation is always higher than {}".format(max_rot)
        # Highest shift that has rotation below max_rot thresold
        final_shift = np.where(rot_valid[-1 - shift:])[0][-1]
        return self.get_frame(final_shift)

    @timer_decorator
    def get_previous_frames(self, shifts=[1], displacements=None, max_rot=1):
        if displacements is not None:
            frames = zip(*[self.get_previous_frame(displacement=d, max_rot=max_rot) for d in displacements])
        else:
            frames = zip(*[self.get_previous_frame(shift=s, max_rot=max_rot) for s in shifts])
        return frames


def inference_toolkit_example():
    parser = ArgumentParser(description='Example usage of Inference toolkit',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_root', metavar='DIR', type=Path)
    parser.add_argument('--depth_output', metavar='FILE', type=Path,
                        help='where to store the estimated depth maps, must be a npy file')
    parser.add_argument('--evaluation_list_path', metavar='PATH', type=Path,
                        help='File with list of images to test for depth evaluation')
    parser.add_argument('--scale-invariant', action='store_true',
                        help='If selected, will rescale depth map with ratio of medians')
    args = parser.parse_args()

    with open(args.evaluation_list_path) as f:
        evaluation_list = [line[:-1] for line in f.readlines()]

    def my_model(frame, previous, pose):
        # Mock up function that uses two frames and translation magnitude
        # return frame[..., -1]
        return np.linalg.norm(pose[:, -1]) * np.linalg.norm(frame - previous, axis=-1)
        # return np.exp(np.random.randn(frame.shape[0], frame.shape[1]))

    engine = inferenceFramework(args.dataset_root, evaluation_list, lambda x: x.transpose(2, 0, 1).astype(np.float32)[None]/255)
    estimated_depth_maps = {}
    mean_time = []
    for sample, image_path in zip(engine, tqdm(evaluation_list)):
        latest_frame, latest_intrinsics, _ = sample.get_frame()
        previous_frame, previous_intrinsics, previous_pose = sample.get_previous_frame(displacement=0.3)
        estimated_depth_maps[image_path] = (my_model(latest_frame, previous_frame, previous_pose))
        time_spent = engine.finish_frame(sample)
        mean_time.append(time_spent)

    print("Mean time per sample : {:.2f}us".format(1e6 * sum(mean_time)/len(mean_time)))
    np.savez(args.depth_output, **estimated_depth_maps)


if __name__ == '__main__':
    inference_toolkit_example()
