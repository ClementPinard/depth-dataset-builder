from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
from imageio import imread, imwrite
from skimage.transform import rescale
from skimage.measure import block_reduce
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tqdm import tqdm
from wrappers import FFMpeg
import gzip
from pebble import ProcessPool


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
    return depth_viz*255


def process_one_frame(img_path, depth_path, occ_path, output_dir, downscale):
    img = imread(img_path)
    h, w, _ = img.shape
    assert((h/downscale).is_integer() and (w/downscale).is_integer())
    output_img = np.zeros((2*(h//downscale), 2*(w//downscale), 3), dtype=np.uint8)
    output_img[:h//downscale, :w//downscale] = rescale(img, 1/downscale, multichannel=True)*255
    if depth_path is not None:
        with gzip.open(depth_path, "rb") as f:
            depth = np.frombuffer(f.read(), np.float32).reshape(h, w)
        output_img[:h//downscale, w//downscale:] = apply_cmap_and_resize(depth, 'rainbow', downscale)
        output_img[h//downscale:, :w//downscale] = \
            output_img[:h//downscale, :w//downscale]//2 + \
            output_img[:h//downscale, w//downscale:]//2

    if occ_path is not None:
        with gzip.open(occ_path, "rb") as f:
            occ = np.frombuffer(f.read(), np.float32).reshape(h, w)
        output_img[h//downscale:, w//downscale:] = apply_cmap_and_resize(occ, 'bone', downscale)

    imwrite(output_dir/img_path.namebase + '.png', output_img)


parser = ArgumentParser(description='create a vizualisation from ground truth created',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--depth_dir', metavar='DIR', type=Path)
parser.add_argument('--img_dir', metavar='DIR', type=Path)
parser.add_argument('--occ_dir', metavar='DIR', type=Path)
parser.add_argument('--output_dir', metavar='DIR', default=None, type=Path)
parser.add_argument('--video', action='store_true')
parser.add_argument('--fps', default='1')
parser.add_argument('--downscale', type=int, default=1)


def process_viz(depth_dir, img_dir, occ_dir, output_dir, video, fps, downscale, ffmpeg, threads=8, **env):
    imgs = sorted(img_dir.files('*.jpg')) + sorted(img_dir.files('*.JPG'))
    depth_maps = []
    occ_maps = []
    for i in imgs:
        fname = i.basename()
        depth_path = depth_dir / fname + ".gz"
        if depth_path.isfile():
            depth_maps.append(depth_path)
        else:
            print("Image {} was not registered".format(fname))
            depth_maps.append(None)
        occ_path = occ_dir / fname + ".gz"
        if occ_path.isfile():
            occ_maps.append(occ_path)
        else:
            occ_maps.append(None)
    output_dir.makedirs_p()
    if threads == 1:
        for i, d, o in tqdm(zip(imgs, depth_maps, occ_maps), total=len(imgs)):
            process_one_frame(i, d, o, output_dir, downscale)
    else:
        with ProcessPool(max_workers=threads) as pool:
            tasks = pool.map(process_one_frame, imgs, depth_maps, occ_maps, [output_dir]*len(imgs), [downscale]*len(imgs))
            try:
                for _ in tqdm(tasks.result(), total=len(imgs)):
                    pass
            except KeyboardInterrupt as e:
                tasks.cancel()
                raise e

    if video:
        video_path = str(output_dir/'video.mp4')
        glob_pattern = str(output_dir/'*.png')
        ffmpeg.create_video(video_path, glob_pattern, fps)


if __name__ == '__main__':
    args = parser.parse_args()
    env = vars(args)
    env["ffmpeg"] = FFMpeg()
    process_viz(**env)
