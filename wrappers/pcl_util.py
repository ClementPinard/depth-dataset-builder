from .default_wrapper import Wrapper


class PCLUtil(Wrapper):

    def __init__(self, build_folder, quiet=False):
        super().__init__(None, quiet)
        self.build_folder = build_folder

    def __call__(self, options):
        self.binary = self.build_folder / options[0]
        super().__call__(options[1:])

    def subsample(self, input_file, output_file, resolution=0.05):
        options = ["PointCloudSubsampler", "--point_cloud_path", input_file,
                   "--resolution", str(resolution), "--output", output_file]
        self.__call__(options)

    def triangulate_mesh(self, output_file, input_file, resolution=0.2):
        options = ["MeshTriangulator", "--point_normal_cloud_path", input_file,
                   "--resolution", str(resolution), "--out_mesh", output_file]
        self.__call__(options)
