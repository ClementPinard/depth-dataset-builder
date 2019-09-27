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
    padded = torch.nn.functional.pad(mask.reshape(1, 1, *mask.shape), [width]*4, value=1)
    filtered = torch.nn.functional.conv2d(padded, kernel)
    mask = (filtered == 1).float()

    return mask


@torch.no_grad()
def extract_sky_mask(network, image_path, mask_folder):
    image = imageio.imread(image_path)
    h, w, _ = image.shape
    image_tensor = torch.from_numpy(image).float()/255
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # shape [1, C, H, W]

    scale_factor = 512/image_tensor.shape[2]
    reduced = F.interpolate(image_tensor, scale_factor=scale_factor, mode='area')

    result = network(reduced.cuda())
    classes = torch.max(result[0], 0)[1]
    mask = (classes == sky_index).float()

    filtered_mask = erosion(1, mask)

    upsampled = F.interpolate(filtered_mask, size=(h, w), mode='nearest')

    final_mask = 1 - upsampled[0].permute(1, 2, 0).cpu().numpy()

    imageio.imwrite(mask_folder/(image_path.basename() + '.png'), (final_mask*255).astype(np.uint8))


parser = ArgumentParser(description='sky mask generator using ENet trained on cityscapes',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--root', metavar='DIR', default="~/Images/scan_manoir",
                    help='path to image folder root')

if __name__ == '__main__':
    args = parser.parse_args()
    network = prepare_network()
    if args.root[-1] == "/":
        args.root = args.root[:-1]
    root = Path(args.root).expanduser()
    mask_root = root + '_mask'
    mask_root.mkdir_p()
    folders = [root] + list(root.walkdirs())
    file_exts = ['jpg', 'JPG']

    for folder in folders:

        mask_folder = mask_root/root.relpathto(folder)
        mask_folder.mkdir_p()
        images = sum((folder.files('*{}'.format(ext)) for ext in file_exts), [])
        if images:
            print("Generating masks for images in {}".format(str(folder)))
            for image_path in tqdm(images):
                extract_sky_mask(network, image_path, mask_folder)
