from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
from imageio import imread, imwrite
import numpy as np
from colmap import read_model as rm
from skimage.transform import rescale
from skimage.measure import block_reduce

parser = ArgumentParser(description='create a vizualisation from ground truth created',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--img_root', metavar='DIR', type=Path)
parser.add_argument('--depth_dir', metavar='DIR', type=Path)
parser.add_argument('--input_model', metavar='DIR', type=Path)
parser.add_argument('--output_dir', metavar='DIR', default=None, type=Path)
parser.add_argument('--downscale', type=int, default=1)


def save_intrinsics(cam, output_dir):
    assert('PINHOLE' in cam.model)
    if 'SIMPLE' in cam.model:
        fx, cx, cy = cam.params
        fy = fx
    else:
        fx, fy, cx, cy = cam.params

    intrinsics = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
    np.savetxt(output_dir/'intrinsics.txt', intrinsics)


def save_depth_maps(cam, depth_maps, output_dir, downscale=1):
    h, w = cam.width, cam.height
    for depth_path in depth_maps:
        depth = np.fromfile(depth_path, np.float32).reshape(h, w)
        downscale_depth = block_reduce(depth, (downscale, downscale), np.min)
        depth_name = depth_path.namebase + '.npy'
        np.save(output_dir / depth_name, downscale_depth)


def save_imgs(img_root, images, depth_maps, output_dir, downscale=1):
    for _, img in images.items():
        img_path = img_root/img.name
        image = rescale(imread(img_path), 1/downscale, multichannel=True)*255
        imwrite(output_dir/img_path.basename(), image.astype(np.uint8))


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


def main():
    args = parser.parse_args()
    cameras, images, _ = rm.read_model(args.input_model, '.txt')
    assert(len(cameras) == 1)
    cam = cameras[list(cameras.keys())[0]]
    save_intrinsics(cam, args.output_dir)
    depth_maps = []
    for key, i in images.items():
        fname = Path(i.name).basename()
        depth_path = args.depth_dir / fname
        if depth_path.isfile():
            depth_maps.append(depth_path)
        else:
            print("Image {} was not registered".format(fname))
            images[key] = None
    save_depth_maps(cam, depth_maps, args.output_dir, args.downscale)
    save_imgs(args.img_root, images, depth_maps, args.output_dir, args.downscale)
    save_positions(images, args.output_dir)


if __name__ == '__main__':
    main()
