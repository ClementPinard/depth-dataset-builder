import torch
import torch.nn.functional as F
import imageio
from model.enet import ENet
from path import Path
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

cityscapes_labels = ['unlabeled', 'road', 'sidewalk',
                     'building', 'wall', 'fence', 'pole',
                     'traffic_light', 'traffic_sign', 'vegetation',
                     'terrain', 'sky', 'person', 'rider', 'car',
                     'truck', 'bus', 'train', 'motorcycle', 'bicycle']

sky_index = cityscapes_labels.index('sky')


def prepare_network():
    ENet_model = ENet(len(cityscapes_labels))
    checkpoint = torch.load('model/ENet')
    ENet_model.load_state_dict(checkpoint['state_dict'])
    return ENet_model.eval().cuda()


def erosion(width, mask):
    kernel = torch.ones(1, 1, 2 * width + 1, 2 * width + 1).to(mask) / (2 * width + 1)**2
    padded = torch.nn.functional.pad(mask.unsqueeze(1), [width]*4, value=1)
    filtered = torch.nn.functional.conv2d(padded, kernel)
    mask = (filtered == 1).float()

    return mask


@torch.no_grad()
def extract_sky_mask(network, image_paths, mask_folder):
    images = np.stack([imageio.imread(i) for i in image_paths])
    if len(images.shape) == 3:
        images = np.stack(3 * [images], axis=-1)
    b, h, w, _ = images.shape
    image_tensor = torch.from_numpy(images).float()/255
    image_tensor = image_tensor.permute(0, 3, 1, 2)  # shape [B, C, H, W]

    w_r = 512
    h_r = int(512 * h / w)
    reduced = F.interpolate(image_tensor, size=(h_r, w_r), mode='area')

    result = network(reduced.cuda())
    classes = torch.max(result, 1)[1]
    mask = (classes == sky_index).float()

    filtered_mask = erosion(1, mask)
    upsampled = F.interpolate(filtered_mask, size=(h, w), mode='nearest')

    final_masks = 1 - upsampled.permute(0, 2, 3, 1).cpu().numpy()

    for f, path in zip(final_masks, image_paths):
        imageio.imwrite(mask_folder/(path.basename() + '.png'), (f*255).astype(np.uint8))


def process_folder(folder_to_process, colmap_img_root, mask_path, pic_ext, verbose=False, batchsize=8, **env):
    network = prepare_network()
    folders = [folder_to_process] + list(folder_to_process.walkdirs())
    for folder in folders:

        mask_folder = mask_path/colmap_img_root.relpathto(folder)
        mask_folder.makedirs_p()
        images = sum((folder.files('*{}'.format(ext)) for ext in pic_ext), [])
        if images:
            if verbose:
                print("Generating masks for images in {}".format(str(folder)))
                images = tqdm(images)
            to_process = []
            for image_file in images:
                if (mask_folder / (image_file.basename() + '.png')).isfile():
                    continue
                to_process.append(image_file)
                if len(to_process) == batchsize:
                    extract_sky_mask(network, to_process, mask_folder)
                    to_process = []
            if to_process:
                extract_sky_mask(network, to_process, mask_folder)
    del network
    torch.cuda.empty_cache()

parser = ArgumentParser(description='sky mask generator using ENet trained on cityscapes',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--img_dir', metavar='DIR', default=Path("workspace/Pictures"),
                    help='path to image folder root', type=Path)
parser.add_argument('--colmap_img_root', metavar='DIR', default=Path("workspace/Pictures"), type=Path,
                    help='image_path you will give to colmap when extracting feature')
parser.add_argument('--mask_root', metavar='DIR', default=Path("workspace/Masks"),
                    help='where to store the generated_masks', type=Path)
parser.add_argument("--batch_size", "-b", type=int, default=8)

if __name__ == '__main__':
    args = parser.parse_args()
    network = prepare_network()
    if args.img_dir[-1] == "/":
        args.img_dir = args.img_dir[:-1]
    args.mask_root.makedirs_p()
    file_exts = ['jpg', 'JPG']

    process_folder(args.img_dir, args.colmap_img_root, args.mask_root, file_exts, True, args.batchsize)

