import numpy as np
from path import Path
from imageio import imread
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class Timer:
    """
    Timer class is used to measure elapsed time, while being able to
    pause it when needed. This is useful to measure algorithm inference
    time without measuring time spent retrieving wanted images
    """
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
    """Inference Framework used for simulating navigation conditions
    for depth algorithms on a dataset created by RDC. It also comes with a way to measure your inference time
    and to record your estimated depths.
    The framework is iterable, and each iteration gives an Inference Sample Object from which you can get images
    to compute depth on.

    Attributes:
        root (Path): Root directory where the Final output of RDC is stored.

        max_shift (float): Max number of frames the algorithm is allowed to search in the past. If the algorithm
        eg. wants to get a frame that was at a particular distance from the last frame, with a barely moving camera,
        the frame can only be as anterior as {max_shift} frames before, even if it means the movement won't be enough.

        estimated_depth_maps (dict): Dictionnary for estimated depth maps, as numpy arrays. Key is image path
        of image on which we estimated depth.

        inference_time (List): List of time spent by your algorithm for inference.
        Will be used at the end of the evaluation to compute the mean inference time

        frame_transform (function): function which will be used to transform images
        before returning them to the algorithm. The function takes a numpy array as
        an argument and can return anything your algorithm want, eg. a pytorch tensor.
    """
    def __init__(self, root, test_files, max_shift=50, frame_transform=None):
        self.root = Path(root)
        self.test_files = test_files
        self.max_shift = max_shift
        self.frame_transform = frame_transform
        self.inference_time = []
        self.estimated_depth_maps = {}

    def __getitem__(self, i):
        """Get item routine. Before returning the sample, the timer is triggered to measure inference time.

        Args:
            i (int): Position of the sample in the test_files list, which has been created with RDC

        Returns:
            InferenceSample: Object to compute depth
        """
        timer = Timer()
        self.i = i
        self.current_sample = inferenceSample(self.root, self.test_files[i], self.max_shift, timer, self.frame_transform)
        self.current_sample.timer.start()
        return self.current_sample

    def finish_frame(self, estimated_depth):
        """Finish Frame routine: This method needs to be called each time your algorithm has
        finished the depth inference. It also stops the timer and stores the time elapsed for this
        sample to compute a mean inference time at the end of the evaluation.

        Args:
            estimated_depth (np.array): The output of your depth algorithm. It will then be stored in
            a dict, and then saved after when it will be completely populated.

        Returns:
            float: time elapsed for inference for this sample
        """
        self.current_sample.timer.stop()
        elapsed = self.current_sample.timer.get_elapsed()
        self.inference_time.append(elapsed)
        self.estimated_depth_maps[self.current_sample.file] = estimated_depth
        return elapsed

    def finalize(self, output_path=None):
        """Finalize: this methods needs to be called at the end of the whole evaluation,
        when there is no sample left to estimate depth on.

        Args:
            output_path (Path, optional): Where to save all the estimated depth. It will
            be saved in a compressed numpy file.

        Returns:
            (float, dict): Return the mean inference time and the compute depth maps in a dictionnary
        """
        if output_path is not None:
            np.savez(output_path, **self.estimated_depth_maps)
        mean_inference_time = np.mean(self.inference_time)
        return mean_inference_time, self.estimated_depth_maps

    def __len__(self):
        return len(self.test_files)


class inferenceSample(object):
    """Inferance Sample class. Is used to get a particular frame with displacement constraints
    For example, you can take the last frame (of which you need to compute the depth map),
    and then want the frame that was 0.3 meters from the last one to ensure a sufficient parallax

    Attributes:
        root (Path): Same as inferenceFramework. Root directory where the Final output of RDC is stored.

        file (Path): image path of image of which we want to estimate depth.

        frame_transform (function) : Same as InferenceFramework. function used to transform loaded image
        into the data format of your choice.

        timer (Timer): timer used to measure time spent computing depth. All the frame gathering and transformation
        are not taken into account in order to only measure inference time.

        valid_frames (List of Path): Ordered list of frame paths representing the frame sequence that is going
        to be used to get the optimal frame pair/set for the algotihm you want to evaluate.
        The order is descending: last frame is first and oldest frames are last.

        poses (np.array): Array of all the poses of the valid_frames list in the R,T format (3x4 matrix).
        They are computed relative to the last frame, and as such, first pose is identity

        rotation_angles (1D np.array): computed from poses, the angle magnitude between last frame and any given frame.
        This is useful when you don't want rotation to be too large.

        displacement (1D np.array): compute from poses, displacement magnitude between last frame and any given frame.
        Useful when you don't want frames to be too close to each other.

        intrinsics (np.array): Intrinsics for each frame, stored in a 3x3 matrix.
    """
    def __init__(self, root, file, max_shift, timer, frame_transform=None):
        self.root = root
        self.file = file
        self.frame_transform = frame_transform
        self.timer = timer
        full_filepath = self.root / file
        scene = full_filepath.parent
        # Get all frames in the scene folder. Normally, there should be more than "max_shift" frames.
        scene_metadata = pd.read_csv(scene : "metadata_.csv")
        scene_files = secene_metadata["image_path"].values
        poses = np.genfromtxt(scene / "poses.txt").reshape((-1, 3, 4))
        sample_id = scene_files.index(full_filepath)
        assert(sample_id > max_shift)
        start_id = sample_id - max_shift
        # Get all frames between start_id (oldest frame) and sample_id.
        # Revert the list so that oldest frames are in the end, like in a buffer
        self.valid_frames = scene_files[start_id:sample_id + 1][::-1]
        # Flip_ud is equivalent to reverting the row and thus the same as [::-1]
        valid_poses = np.flipud(poses[start_id:sample_id + 1])
        # All poses in the sequence should be valid
        assert not np.isnan(valid_poses.sum())
        # Change the pose array so that instead of 3x4 matrices, we have 4x4 matrices, which we can invert
        last_line = np.broadcast_to(np.array([0, 0, 0, 1]), (valid_poses.shape[0], 1, 4))
        valid_poses_full = np.concatenate([valid_poses, last_line], axis=1)
        self.poses = (np.linalg.inv(valid_poses_full[0]) @  valid_poses_full)[:, :3]
        R = self.poses[:,:3,:3]
        self.rotation_angles = Rotation.from_matrix(R).magnitude()
        self.displacements = np.linalg.norm(self.poses[:, :, -1], axis=-1)

        # Case 1 for intrinsics : Zoom level never changed and thus there's only one intrinsics
        # matrix for the whole video, stored in intrinsics.txt This is the most usual case
        # Case 2 : Each frame has its own intrinsics file <name>_intrinsics.txt
        # Case is only here for later compatibility, but it has not been tested thoroughly
        if (scene / "intrinsics.txt").isfile():
            self.intrinsics = np.stack([np.genfromtxt(scene / "intrinsics.txt")] * max_shift)
        else:
            intrinsics_files = [f.stripext() + "_intrinsics.txt" for f in self.valid_frames]
            self.intrinsics = np.stack([np.genfromtxt(i) for i in intrinsics_files])

    def timer_decorator(func, *args, **kwargs):
        """
        Decorator used to pause the timer and only restart it when returning the result.
        This is used to not penalize the inference algorithm when frame retrieving is slow,
        because in real conditions, it's possible you get the wanted frames immediately instead
        of searching for them in the memory.
        """
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
        """Basic function to get frame within a fixed shift. When used without parameters, it returns
        the sample frame.

        Args:
            shift (int, optional): Position relative to sample frame of the frame we want to get.
            Defaults to 0.

        Returns a tuple of 3:
            [Unknown type]: Output of the frame_transform function, used on the desired frame, loaded in a np array
            np.array: 3x3 intrinsics matrix of returned frame
            np.array: 3x4 pose matrix of returned frame
        """
        file = self.valid_frames[shift]
        img = imread(file)
        if self.frame_transform is not None:
            img = self.frame_transform(img)
        return img, self.intrinsics[shift], self.poses[shift]

    @timer_decorator
    def get_previous_frame(self, shift=1, displacement=None, max_rot=1):
        """More advanced function, to get a frame within shift, displacement and rotation constraints. Timer is paused when this
        function is running.

        Args:
            shift (int, optional): As above. Position relative to sample frame of the frame we want to get. Defaults to 1.
            displacement (Float, optional): Desired displacement (in meters) between sample frame and the frame we want to get. This parameter
            overwrite the shift parameter. Defaults to None.
            max_rot (int, optional): Maximum Rotation, in radians. The function cannot return a frame with a higher rotation than max_rot. It assumes
            rotation is growing with time (only true for the first frames). The maximum shift of the returned frame corresponds to the first frame
            with a rotation above this threshold. Defaults to 1.

        Returns a tuple of 3:
            [Unknow type]: Output of the frame_transform function, used on the frame that best represent the different constrains.
            np.array: 3x3 intrinsics matrix of returned frame
            np.array: 3x4 pose matrix of returned frame
        """
        if displacement is not None:
            shift = max(1, np.abs(self.displacements - displacement).argmin())
        rot_valid = self.rotation_angles < max_rot
        assert sum(rot_valid[1:shift+1] > 0), "Rotation is always higher than {}".format(max_rot)
        # Highest shift that has rotation below max_rot thresold
        final_shift = np.where(rot_valid[-1 - shift:])[0][-1]
        return self.get_frame(final_shift)

    @timer_decorator
    def get_previous_frames(self, shifts=[1], displacements=None, max_rot=1):
        """Helper function to get multiple frames at the same time. with the previous function.

        Args:
            shifts (List): list of wanted shifts
            displacements (List): List of wanted displacements, overwrite shifts
            max_rot (int, optional): Maximum Rotation, see previous function

        Returns a tuple of 3:
            List: Outputs of the frame_transform function for each desired frame
            List: 3x3 intrinsics matrices of returned frames
            List: 3x4 pose matrices of returned frames
        """
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
        # Replace it with your algorithm, eg. DepthNet model
        return np.linalg.norm(pose[:, -1]) * np.linalg.norm(frame - previous, axis=-1)

    # This is our transform function. It converts the uint8 array into a float array,
    # divides it by 255 to have values in [0,1] and adds the batch dimensions
    def my_transform(img):
        return x.transpose(2, 0, 1).astype(np.float32)[None] / 255

    engine = inferenceFramework(args.dataset_root, evaluation_list, my_transform)
    for sample in tqdm(engine):
        latest_frame, latest_intrinsics, _ = sample.get_frame()
        previous_frame, previous_intrinsics, previous_pose = sample.get_previous_frame(displacement=0.3)
        engine.finish_frame(my_model(latest_frame, previous_frame, previous_pose))

    mean_time, _ = engine.finalize(args.depth_output)
    print("Mean time per sample : {:.2f}us".format(1e6 * mean_time))


if __name__ == '__main__':
    inference_toolkit_example()
