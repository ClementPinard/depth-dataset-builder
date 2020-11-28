import numpy as np
from path import Path
from imageio import imread
import time


class Timer:
    def __init__(self):
        self._start_time = None
        self._elapsed_time = 0

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
    def __init__(self, root, test_files, seq_length=3, min_depth=1e-3, max_depth=80, max_shift=50):
        self.root = root
        self.test_files = test_files
        self.min_depth, self.max_depth = min_depth, max_depth
        self.max_shift = max_shift

    def __getitem__(self, i):
        timer = Timer()
        sample = inferenceSample(self.root, self.test_files[i], timer, self.max_shift)
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
        full_filepath = self.root / file
        scene = full_filepath.parent
        scene_files = sorted(scene.files("*jpg"))
        poses = np.genfromtxt(scene / "poses.txt").reshape((-1, 3, 4))
        sample_id = scene_files.index(full_filepath)
        assert(sample_id > max_shift)
        start_id = sample_id - max_shift
        self.valid_frames = scene_files[start_id:sample_id + 1][::-1]
        valid_poses = poses[start_id:sample_id + 1].flipud()
        valid_poses_full = np.concatenate([valid_poses, np.array([0, 0, 0, 1]).reshape(1, 4, 1)])
        self.poses = (np.linalg.inv(valid_poses_full[0]) @  valid_poses_full)[:, :3]
        R = self.poses[:, :3, :3]
        s = np.linalg.norm(np.stack([R[:, 0, 1]-R[:, 1, 0],
                                     R[:, 1, 2]-R[:, 2, 1],
                                     R[:, 0, 2]-R[:, 2, 0]]), axis=1)
        self.rotation_angles = np.abs(np.arcsin(0.5 * s))
        self.displacements = np.linalg.norm(self.poses[:, :, 4])

        if (scene / "intrinsics.txt").isfile():
            self.intrinsics = np.stack([np.genfromtxt(scene / "intrinsics.txt")]*max_shift)
        else:
            intrinsics_files = [f.stripext() + "_intrinsics.txt" for f in self.valid_frames]
            self.intrinsics = np.stack([np.genfromtxt(i) for i in intrinsics_files])

    def get_frame(self, shift=0):
        self.timer.stop()
        file = self.valid_frames[shift]
        img = imread(file)
        if self.frame_transform is not None:
            img = self.frame_transform(img)
        self.timer.start()
        return img, self.intrinsics[shift]

    def get_previous_frame(self, shift=1, displacement=None, max_rot=1):
        self.timer.stop()

        if displacement is not None:
            shift = (self.poses[:, :, -1] - displacement).argmin()

        rot_valid = self.rotation_angles < max_rot
        assert sum(rot_valid[1:shift] > 0), "Rotation is alaways higher than {}".format(max_rot)
        # Highest shift that has rotation below max_rot thresold

        final_shift = np.where(rot_valid[-1 - shift:])[-1]
        self.timer.start()
        return *self.get_frame(final_shift), self.poses[final_shift]
