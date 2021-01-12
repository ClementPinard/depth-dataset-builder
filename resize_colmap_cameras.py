from colmap_util import read_model as rm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path

parser = ArgumentParser(description='Resize cameras in a colmap model',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('-i', '--input_cameras_colmap', metavar='FILE', type=Path, required=True,
                    help='Input COLMAP cameras.bin or caameras.txt file to rescale or resize.')
parser.add_argument('-o', '--output_cameras_colmap', metavar='FILE', type=Path, required=True,
                    help='Output images.bin or images.txt file with filtered frame localizations')
im_resize = parser.add_mutually_exclusive_group()
im_resize.add_argument('-w', '--width', type=float, default=None,
                       help='Output width of cameras every camera will have, '
                            'even though they are initially of different size. '
                            'Height reflect initial ratio')
im_resize.add_argument('-r', '--rescale', type=float, default=1,
                       help='float to which each camera dimension will be multiplied. '
                            'As such, cameras ay have different widths')


def resize_cameras(input_cameras, output_cameras, output_width=None, output_rescale=1):
    if input_cameras.ext == ".txt":
        cameras = rm.read_images_text(input_cameras)
    elif input_cameras.ext == ".bin":
        cameras = rm.read_images_binary(input_cameras)
    else:
        print(input_cameras.ext)
    cameras_rescaled = {}
    for i, c in cameras.items():
        if output_width is not None:
            output_rescale = output_width / c.width
        output_width = c.width * output_rescale
        output_height = c.height * output_rescale
        output_params = c.params
        single_focal = ('SIMPLE' in c.model) or ('RADIAL' in c.model)
        output_params[:3] *= output_rescale
        if not single_focal:
            output_params[3] *= output_rescale

        cameras_rescaled[i] = rm.Camera(id=c.id, model=c.model,
                                        width=output_width, height=output_height,
                                        params=c.params)

    rm.write_cameras_text(cameras_rescaled, output_cameras)


if __name__ == '__main__':
    args = parser.parse_args()
    resize_cameras(input_cameras=args.input_cameras_colmap,
                   output_cameras=args.output_cameras_colmap,
                   output_width=args.output_width,
                   output_rescale=args.output_rescale)
