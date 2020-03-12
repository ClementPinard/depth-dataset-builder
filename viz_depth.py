from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
from imageio import imread, imwrite
from skimage.transform import rescale
from skimage.measure import block_reduce
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tqdm import tqdm
from subprocess import Popen, PIPE


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
        depth_norm = downscale_depth * 0 + 1

    depth_viz = COLORMAPS[colormap](depth_norm)[:, :, :3]
    depth_viz[downscale_depth == np.inf] = 0
    return depth_viz*255


parser = ArgumentParser(description='create a vizualisation from ground truth created',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--depth_dir', metavar='DIR', type=Path)
parser.add_argument('--img_dir', metavar='DIR', type=Path)
parser.add_argument('--occ_dir', metavar='DIR', type=Path)
parser.add_argument('--output_dir', metavar='DIR', default=None, type=Path)
parser.add_argument('--video', action='store_true')
parser.add_argument('--fps', default='1')
parser.add_argument('--downscale', type=int, default=1)


def main():
    args = parser.parse_args()
    imgs = sorted(args.img_dir.files('*.jpg')) + sorted(args.img_dir.files('*.JPG'))
    depth_maps = []
    occ_maps = []
    for i in imgs:
        fname = i.basename()
        depth_path = args.depth_dir / fname
        if depth_path.isfile():
            depth_maps.append(depth_path)
        else:
            print("Image {} was not registered".format(fname))
            depth_maps.append(None)
        occ_path = args.occ_dir / fname
        if occ_path.isfile():
            occ_maps.append(occ_path)
        else:
            occ_maps.append(None)
    args.output_dir.makedirs_p()
    for i, d, o in tqdm(zip(imgs, depth_maps, occ_maps), total=len(imgs)):
        img = imread(i)
        h, w, _ = img.shape
        assert((h/args.downscale).is_integer() and (w/args.downscale).is_integer())
        output_img = np.zeros((2*(h//args.downscale), 2*(w//args.downscale), 3), dtype=np.uint8)
        output_img[:h//args.downscale, :w//args.downscale] = rescale(img, 1/args.downscale, multichannel=True)*255
        if d is not None:
            depth = np.fromfile(d, np.float32).reshape(h, w)
            output_img[:h//args.downscale, w//args.downscale:] = apply_cmap_and_resize(depth, 'rainbow', args.downscale)
            output_img[h//args.downscale:, :w//args.downscale] = \
                output_img[:h//args.downscale, :w//args.downscale]//2 + \
                output_img[:h//args.downscale, w//args.downscale:]//2

        if o is not None:
            occ = np.fromfile(o, np.float32).reshape(h, w)
            output_img[h//args.downscale:, w//args.downscale:] = apply_cmap_and_resize(occ, 'bone', args.downscale)

        imwrite(args.output_dir/i.namebase + '.png', output_img)

    if args.video:
        video_path = str(args.output_dir/'video.mp4')
        glob_pattern = str(args.output_dir/'*.png')
        ffmpeg = Popen(["ffmpeg", "-y", "-r", args.fps,
                        "-pattern_type", "glob", "-i",
                        glob_pattern, video_path],
                       stdout=PIPE, stderr=PIPE)
        ffmpeg.wait()


if __name__ == '__main__':
    main()
