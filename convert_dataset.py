from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
from imageio import imread, imwrite
from skimage.transform import rescale
from skimage.measure import block_reduce
from colmap_util import read_model as rm
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tqdm import tqdm
from wrappers import FFMpeg
import gzip
from pebble import ProcessPool
import pandas as pd


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
            intrinsics_name = output_dir / Path(img.name).namebase + "_intrinsics.txt"
            np.savetxt(intrinsics_name, intrinsics)


def to_transform_matrix(q, t):
    cam_R = rm.qvec2rotmat(q).T
    cam_t = (- cam_R @ t).reshape(3, 1)
    transform = np.vstack((np.hstack([cam_R, cam_t]), [0, 0, 0, 1]))
    return transform


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


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def apply_cmap_and_resize(depth, colormap, downscale):
    downscale_depth = block_reduce(depth, (downscale, downscale), np.min)
    finite_depth = depth[depth < np.inf]
    if finite_depth.size != 0:
        max_d = depth[depth < np.inf].max()
        depth_norm = downscale_depth/max_d
        depth_norm[downscale_depth == np.inf] = 1
    else:
        depth_norm = np.ones_like(downscale_depth)

    depth_viz = COLORMAPS[colormap](depth_norm)[:, :, :3]
    depth_viz[downscale_depth == np.inf] = 0
    return downscale_depth, depth_viz*255


def process_one_frame(img_path, depth_path, occ_path,
                      dataset_output_dir, video_output_dir, downscale, interpolated):
    img = imread(img_path)
    h, w, _ = img.shape
    assert((h/downscale).is_integer() and (w/downscale).is_integer())
    output_img = np.zeros((2*(h//downscale), 2*(w//downscale), 3), dtype=np.uint8)
    rescaled_img = rescale(img, 1/downscale, multichannel=True)*255
    # Img goes to upper left corner of vizualisation
    output_img[:h//downscale, :w//downscale] = rescaled_img
    imwrite(dataset_output_dir / img_path.basename(), rescaled_img.astype(np.uint8))
    if depth_path is not None:
        with gzip.open(depth_path, "rb") as f:
            depth = np.frombuffer(f.read(), np.float32).reshape(h, w)
        output_depth_name = dataset_output_dir / img_path.basename() + '.npy'
        downscaled_depth, viz = apply_cmap_and_resize(depth, 'rainbow', downscale)
        if not interpolated:
            np.save(output_depth_name, downscaled_depth)
        # Depth colormap goes to upper right corner
        output_img[:h//downscale, w//downscale:] = viz
        # Mix Depth / image goest to lower left corner
        output_img[h//downscale:, :w//downscale] = \
            output_img[:h//downscale, :w//downscale]//2 + \
            output_img[:h//downscale, w//downscale:]//2

    if occ_path is not None:
        with gzip.open(occ_path, "rb") as f:
            occ = np.frombuffer(f.read(), np.float32).reshape(h, w)
        _, occ_viz = apply_cmap_and_resize(occ, 'bone', downscale)
        # Occlusion depthmap vizualisation goes to lower right corner
        output_img[h//downscale:, w//downscale:] = occ_viz
    if interpolated:
        output_img[:5] = output_img[-5:] = output_img[:, :5] = output_img[:, -5:] = [255, 128, 0]

    imwrite(video_output_dir/img_path.namebase + '.png', output_img)


parser = ArgumentParser(description='create a vizualisation from ground truth created',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--depth_dir', metavar='DIR', type=Path)
parser.add_argument('--img_dir', metavar='DIR', type=Path)
parser.add_argument('--occ_dir', metavar='DIR', type=Path)
parser.add_argument('--metdata', type=Path)
parser.add_argument('--output_dir', metavar='DIR', default=None, type=Path)
parser.add_argument('--video', action='store_true')
parser.add_argument('--fps', default='1')
parser.add_argument('--downscale', type=int, default=1)


def convert_dataset(final_model, depth_dir, images_root_folder, occ_dir, dataset_output_dir, video_output_dir, metadata_path, interpolated_frames_path,
                    fps, downscale, ffmpeg, threads=8, video=False, **env):
    dataset_output_dir.makedirs_p()
    video_output_dir.makedirs_p()
    cameras, images, _ = rm.read_model(final_model, '.txt')
    save_intrinsics(cameras, images, dataset_output_dir, downscale)
    save_positions(images, dataset_output_dir)
    if interpolated_frames_path is None:
        interpolated_frames = []
    else:
        with open(interpolated_frames_path, "r") as f:
            interpolated_frames = [line[:-1] for line in f.readlines()]

    metadata = pd.read_csv(metadata_path).set_index("db_id", drop=False).sort_values("time")
    image_df = pd.DataFrame.from_dict(images, orient="index").set_index("id")
    image_df = image_df.reindex(metadata.index)
    depth_maps = []
    occ_maps = []
    interpolated = []
    imgs = []
    cameras = []

    for i in metadata["image_path"]:
        img_path = images_root_folder / Path(i).relpath("Videos")
        imgs.append(img_path)

        fname = img_path.basename()
        depth_path = depth_dir / fname + ".gz"
        if depth_path.isfile():
            depth_maps.append(depth_path)
        else:
            print("Image {} was not registered".format(fname))
            depth_maps.append(None)
        if i in interpolated_frames:
            interpolated.append(True)
            print("Image {} was interpolated".format(fname))
        else:
            interpolated.append(False)

        occ_path = occ_dir / fname + ".gz"
        if occ_path.isfile():
            occ_maps.append(occ_path)
        else:
            occ_maps.append(None)
    if threads == 1:
        for i, d, o, n in tqdm(zip(imgs, depth_maps, occ_maps, interpolated), total=len(imgs)):
            process_one_frame(i, d, o, dataset_output_dir, video_output_dir, downscale, n)
    else:
        with ProcessPool(max_workers=threads) as pool:
            tasks = pool.map(process_one_frame, imgs, depth_maps, occ_maps,
                             [dataset_output_dir]*len(imgs), [video_output_dir]*len(imgs),
                             [downscale]*len(imgs), interpolated)
            try:
                for _ in tqdm(tasks.result(), total=len(imgs)):
                    pass
            except KeyboardInterrupt as e:
                tasks.cancel()
                raise e

    if video:
        video_path = str(video_output_dir/'{}_groundtruth_viz.mp4'.format(video_output_dir.namebase))
        glob_pattern = str(video_output_dir/'*.png')
        ffmpeg.create_video(video_path, glob_pattern, fps)


if __name__ == '__main__':
    args = parser.parse_args()
    env = vars(args)
    env["ffmpeg"] = FFMpeg()
    convert_dataset(**env)
