from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
from imageio import imread, imwrite
import numpy as np
from colmap_util import read_model as rm
from skimage.transform import rescale
from skimage.measure import block_reduce
import gzip
from pebble import ProcessPool
from tqdm import tqdm

parser = ArgumentParser(description='create a visualization from ground truth created',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--img_dir', metavar='DIR', type=Path)
parser.add_argument('--depth_dir', metavar='DIR', type=Path)
parser.add_argument('--input_model', metavar='DIR', type=Path)
parser.add_argument('--output_dir', metavar='DIR', default=None, type=Path)
parser.add_argument('--downscale', type=int, default=1)


def save_intrinsics(cameras, images, output_dir, downscale=1):
    def construct_intrinsics(cam):
        assert('PINHOLE' in cam.model)
        if 'SIMPLE' in cam.model:
            fx, cx, cy = cam.params
            fy = fx
        else:
            fx, fy, cx, cy = cam.params

        return np.array([[fx / downscale, 0, cx / downscale],
                         [0, fy / downscale, cy / downscale],
                         [0, 0, 1]])

    if len(cameras) == 1:
        cam = cameras[list(cameras.keys())[0]]
        intrinsics = construct_intrinsics(cam)
        np.savetxt(output_dir / 'intrinsics.txt', intrinsics)
    else:
        for _, img in images.items():
            cam = cameras[img.camera_id]
            intrinsics = construct_intrinsics(cam)
            intrinsics_name = output_dir / Path(img.name).stem + "_intrinsics.txt"
            np.savetxt(intrinsics_name, intrinsics)


def process_one_frame(cameras, img, img_root, depth_dir, output_dir, downscale):
    cam = cameras[img.camera_id]
    img_name = Path(img.name)
    depth_path = depth_dir / img_name.basename() + ".gz"
    h, w = cam.width, cam.height
    with gzip.open(depth_path, "rb") as f:
        depth = np.frombuffer(f.read(), np.float32).reshape(h, w)
    downscaled_depth = block_reduce(depth, (downscale, downscale), np.min)
    output_depth_name = output_dir / img_name.basename() + '.npy'
    np.save(output_depth_name, downscaled_depth)

    input_img_path = img_root / img_name
    output_img_path = output_dir / img_name.basename()
    image = rescale(imread(input_img_path), 1/downscale, multichannel=True)*255
    imwrite(output_img_path, image.astype(np.uint8))


def save_positions(images, output_dir):
    starting_pos = None
    positions = []
    for _, img in images.items():
        current_pos = to_transform_matrix(img.qvec, img.tvec)
        if starting_pos is None:
            starting_pos = current_pos
        relative_position = np.linalg.inv(starting_pos) @ current_pos
        positions.append(relative_position[:3])
    positions = np.stack(positions)
    np.savetxt(output_dir/'poses.txt', positions.reshape((len(images), -1)))


def to_transform_matrix(q, t):
    cam_R = rm.qvec2rotmat(q).T
    cam_t = (- cam_R @ t).reshape(3, 1)
    transform = np.vstack((np.hstack([cam_R, cam_t]), [0, 0, 0, 1]))
    return transform


def convert_to_kitti(input_model, img_dir, depth_dir, output_dir, downscale=1, threads=1, **env):
    cameras, images, _ = rm.read_model(input_model, '.txt')
    save_intrinsics(cameras, images, output_dir, downscale)
    save_positions(images, output_dir)
    if threads == 1:
        for _, img in tqdm(images.items()):
            process_one_frame(cameras, img, img_dir, depth_dir, output_dir, downscale)
    else:
        with ProcessPool(max_workers=threads) as pool:

            tasks = pool.map(process_one_frame, [cameras] * len(images),
                             [img for _, img in images.items()],
                             [img_dir] * len(images),
                             [depth_dir] * len(images),
                             [output_dir] * len(images),
                             [downscale] * len(images))
            try:
                for _ in tqdm(tasks.result(), total=len(images)):
                    pass
            except KeyboardInterrupt as e:
                tasks.cancel()
                raise e


if __name__ == '__main__':
    args = parser.parse_args()
    convert_to_kitti(vars(args))
