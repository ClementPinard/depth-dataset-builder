from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
from imageio import imread, imwrite
from skimage.transform import rescale, resize
from skimage.measure import block_reduce
from colmap_util import read_model as rm
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tqdm import tqdm
from wrappers import FFMpeg
import gzip
from pebble import ProcessPool
import yaml


def save_intrinsics(cameras, images, output_dir, output_width=None, downscale=None):
    def construct_intrinsics(cam, downscale):
        # assert('PINHOLE' in cam.model)
        if 'SIMPLE' in cam.model:
            fx, cx, cy, *_ = cam.params
            fy = fx
        else:
            fx, fy, cx, cy, *_ = cam.params

        return np.array([[fx / downscale, 0, cx / downscale],
                         [0, fy / downscale, cy / downscale],
                         [0, 0, 1]])

    def save_cam(cam, intrinsics_path, yaml_path):
        if downscale is None:
            current_downscale = output_width / cam.width
        else:
            current_downscale = downscale
        intrinsics = construct_intrinsics(cam, current_downscale)
        np.savetxt(intrinsics_path, intrinsics)
        with open(yaml_path, 'w') as f:
            camera_dict = {"model": cam.model,
                           "params": cam.params,
                           "width": cam.width / current_downscale,
                           "height": cam.height / current_downscale}
            yaml.dump(camera_dict, f, default_flow_style=False)

    if len(cameras) == 1:
        print("bonjour")
        cam = cameras[list(cameras.keys())[0]]
        save_cam(cam, output_dir / "intrinsics.txt", output_dir / "camera.yaml")

    else:
        print("au revoir")
        for _, img in images.items():
            cam = cameras[img.camera_id]

            save_cam(cam, output_dir / Path(img.name).stem + "_intrinsics.txt",
                     output_dir / Path(img.name).stem + "_camera.yaml")


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
                      dataset_output_dir, video_output_dir, downscale, interpolated,
                      visualization=False, viz_width=1920, compressed=True):
    img = imread(img_path)
    if len(img.shape) == 3:
        h, w, _ = img.shape
    elif len(img.shape) == 2:
        h, w = img.shape
        img = img.reshape(h, w, 1)
    assert(viz_width % 2 == 0)
    viz_height = int(viz_width * h / (2*w)) * 2
    output_img = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)
    rescaled_img = rescale(img, 1/downscale, multichannel=True)*255
    imwrite(dataset_output_dir / img_path.basename(), rescaled_img.astype(np.uint8))

    if visualization:
        viz_img = resize(img, (viz_height//2, viz_width//2))*255
        # Img goes to upper left corner of visualization
        output_img[:viz_height//2, :viz_width//2] = viz_img
    if depth_path is not None:
        with gzip.open(depth_path, "rb") if compressed else open(depth_path, "rb") as f:
            depth = np.frombuffer(f.read(), np.float32).reshape(h, w)
        output_depth_name = dataset_output_dir / img_path.basename() + '.npy'
        downscaled_depth, viz = apply_cmap_and_resize(depth, 'rainbow', downscale)
        if not interpolated:
            np.save(output_depth_name, downscaled_depth)
        if visualization:
            viz_rescaled = resize(viz, (viz_height//2, viz_width//2))
            # Depth colormap goes to upper right corner
            output_img[:viz_height//2, viz_width//2:] = viz_rescaled
            # Mix Depth / image goest to lower left corner
            output_img[viz_height//2:, :viz_width//2] = \
                output_img[:viz_height//2, :viz_width//2]//2 + \
                output_img[:viz_height//2, viz_width//2:]//2

    if occ_path is not None and visualization:
        with gzip.open(occ_path, "rb") if compressed else open(occ_path, "rb") as f:
            occ = np.frombuffer(f.read(), np.float32).reshape(h, w)
        _, occ_viz = apply_cmap_and_resize(occ, 'bone', downscale)
        occ_viz_rescaled = resize(occ_viz, (viz_height//2, viz_width//2))
        # Occlusion depthmap visualization goes to lower right corner
        output_img[viz_height//2:, viz_width//2:] = occ_viz_rescaled
    if interpolated:
        output_img[:5] = output_img[-5:] = output_img[:, :5] = output_img[:, -5:] = [255, 128, 0]

    if visualization:
        imwrite(video_output_dir/img_path.stem + '.png', output_img)


parser = ArgumentParser(description='Convert dataset to KITTI format, optionnally create a visualization video',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--depth_dir', metavar='DIR', type=Path, required=True,
                    help='folder where depth maps generated by ETH3D are stored Usually ends with  "ground_truth_depth/<video name>"')
parser.add_argument('--images_root_folder', metavar='DIR', type=Path, required=True,
                    help='folder where video frames are stored')
parser.add_argument('--occ_dir', metavar='DIR', type=Path,
                    help='folder where occlusion depth maps generated by ETH3D are stored. Usually ends with "occlusion_depth/<video name>"')
parser.add_argument('--metadata_path', type=Path, required=True,
                    help='path to metadata CSV file generated during video_to_colmap.py')
parser.add_argument('--dataset_output_dir', metavar='DIR', default=None, type=Path, required=True)
parser.add_argument('--video_output_dir', metavar='DIR', default=None, type=Path)
parser.add_argument('--interpolated_frames_path', metavar='TXT', type=Path)
parser.add_argument('--final_model', metavar='DIR', type=Path)
parser.add_argument('--visualize', action='store_true',
                    help='If selected, will generate images with depth colorized for visualization purpose')
parser.add_argument('--video', action='store_true',
                    help='If selected, will generate a video from visualization images')
parser.add_argument('--downscale', type=int, default=1, help='How much ground truth depth is downscaled in order to save space')
parser.add_argument('--threads', '-j', type=int, default=8, help='')
parser.add_argument('--compressed', action='store_true',
                    help='Indicates if GroundTruthCreator was used with option `--compress_depth_maps`')
parser.add_argument('--verbose', '-v', action='count', default=0)


def convert_dataset(final_model, depth_dir, images_root_folder, occ_dir,
                    dataset_output_dir, video_output_dir, ffmpeg,
                    interpolated_frames=[], metadata=None, images_list=None,
                    threads=8, downscale=None, compressed=True,
                    width=None, visualization=False, video=False, verbose=0, **env):
    dataset_output_dir.makedirs_p()
    video_output_dir.makedirs_p()
    if video:
        visualization = True
    cameras, images, _ = rm.read_model(final_model, '.txt')
    # image_df = pd.DataFrame.from_dict(images, orient="index").set_index("id")

    if metadata is not None:
        metadata = metadata.set_index("db_id", drop=False).sort_values("time")
        framerate = metadata["framerate"].values[0]
        # image_df = image_df.reindex(metadata.index)
        images_list = metadata["image_path"]
    else:
        assert images_list is not None
        framerate = None
        video = False

    # Discard images and cameras that are not represented by the image list
    images = {k: i for k, i in images.items() if i.name in images_list}
    cameras_ids = set([i.camera_id for i in images.values()])
    cameras = {k: cameras[k] for k in cameras_ids}

    if downscale is None:
        assert width is not None
    save_intrinsics(cameras, images, dataset_output_dir, width, downscale)
    save_positions(images, dataset_output_dir)

    depth_maps = []
    occ_maps = []
    interpolated = []
    imgs = []
    cameras = []
    not_registered = 0

    for i in images_list:
        img_path = images_root_folder / i
        imgs.append(img_path)

        fname = img_path.basename()
        depth_path = depth_dir / fname
        occ_path = occ_dir / fname
        if compressed:
            depth_path += ".gz"
            occ_path += ".gz"
        if depth_path.isfile():
            if occ_path.isfile():
                occ_maps.append(occ_path)
            else:
                occ_maps.append(None)
            depth_maps.append(depth_path)
            if i in interpolated_frames:
                if verbose > 2:
                    print("Image {} was interpolated".format(fname))
                interpolated.append(True)
            else:
                interpolated.append(False)
        else:
            if verbose > 2:
                print("Image {} was not registered".format(fname))
            not_registered += 1
            depth_maps.append(None)
            occ_maps.append(None)
            interpolated.append(False)
    print('{}/{} Frames not registered ({:.2f}%)'.format(not_registered, len(images_list), 100*not_registered/len(images_list)))
    print('{}/{} Frames interpolated ({:.2f}%)'.format(sum(interpolated), len(images_list), 100*sum(interpolated)/len(images_list)))
    if threads == 1:
        for i, d, o, n in tqdm(zip(imgs, depth_maps, occ_maps, interpolated), total=len(imgs)):
            process_one_frame(i, d, o, dataset_output_dir, video_output_dir, downscale, n, visualization, viz_width=1920)
    else:
        with ProcessPool(max_workers=threads) as pool:
            tasks = pool.map(process_one_frame, imgs, depth_maps, occ_maps,
                             [dataset_output_dir]*len(imgs), [video_output_dir]*len(imgs),
                             [downscale]*len(imgs), interpolated,
                             [visualization]*len(imgs), [1920]*len(imgs))
            try:
                for _ in tqdm(tasks.result(), total=len(imgs)):
                    pass
            except KeyboardInterrupt as e:
                tasks.cancel()
                raise e

    if video:
        video_path = str(video_output_dir.parent/'{}_groundtruth_viz.mp4'.format(video_output_dir.stem))
        glob_pattern = str(video_output_dir/'*.png')
        ffmpeg.create_video(video_path, glob_pattern, True, framerate)
        video_output_dir.rmtree_p()


if __name__ == '__main__':
    args = parser.parse_args()
    env = vars(args)
    if args.interpolated_frames_path is None:
        env["interpolated_frames"] = []
    else:
        with open(args.interpolated_frames_path, "r") as f:
            env["interpolated_frames"] = [line[:-1] for line in f.readlines()]
    env["ffmpeg"] = FFMpeg()
    convert_dataset(**env)
