from .default_wrapper import Wrapper


class PCLUtil(Wrapper):

    def __init__(self, build_folder, logfile=None, quiet=False):
        super().__init__(None, quiet, logfile)
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

    def register_reconstruction(self, georef, lidar, output_matrix, output_cloud=None, max_distance=1):
        options = ["CloudRegistrator", "--georef", georef,
                   "--lidar", lidar, "--max_distance", str(max_distance),
                   "--output_matrix", output_matrix]
        if output_cloud is not None:
            options += ["--output_cloud", output_cloud]
        self.__call__(options)
