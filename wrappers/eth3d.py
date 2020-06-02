from .default_wrapper import Wrapper


class ETH3D(Wrapper):
    """docstring for Colmap"""

    def __init__(self, build_folder, image_path, max_occlusion_depth, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.build_folder = build_folder
        self.image_path = image_path
        self.max_occlusion_depth = max_occlusion_depth

    def __call__(self, options):
        self.binary = self.build_folder / options[0]
        super().__call__(options[1:])

    def align_with_ICP(self, input_model, output_model, scales=5):
        options = ["ICPScanAligner", "-i", input_model,
                   "-o", output_model, "--number_of_scales", str(scales)]
        self.__call__(options)

    def clean_pointcloud(self, input_model, filter=(5, 10)):
        options = ["PointCloudCleaner", "--in", input_model,
                   "--filter", "{},{}".format(*filter)]
        self.__call__(options)

    def compute_normals(self, output_ply, scan_meshlab, neighbor_count=None, neighbor_radius=0.2):
        options = ["NormalEstimator", "-i", scan_meshlab, "-o", output_ply]
        if neighbor_count is not None:
            options += ["--neighbor_count", str(neighbor_count)]
        if neighbor_radius is not None:
            options += ["--neighbor_radius", str(neighbor_radius)]
        self.__call__(options)

    def create_splats(self, output_splats, pointnormals_ply, occlusion_ply, threshold=0.1):
        options = ["SplatCreator", "--point_normal_cloud_path", pointnormals_ply,
                   "--mesh_path", occlusion_ply,
                   "--output_path", output_splats,
                   "--distance_threshold", str(threshold)]
        self.__call__(options)

    def create_ground_truth(self, scan_meshlab, colmap_model, output_folder, occlusions=None, splats=None,
                            point_cloud=True, depth_maps=True, occlusion_maps=True):
        options = ["GroundTruthCreator", "--scan_alignment_path", scan_meshlab,
                   "--image_base_path", self.image_path, "--state_path", colmap_model,
                   "--output_folder_path", output_folder, "--occlusion_mesh_path", occlusions,
                   "--occlusion_splats_path", splats,
                   "--max_occlusion_depth", str(self.max_occlusion_depth),
                   "--write_point_cloud", "1" if point_cloud else "0",
                   "--write_depth_maps", "1" if depth_maps else "0",
                   "--write_occlusion_depth", "1" if occlusion_maps else "0",
                   "--compress_depth_maps", "1"]
        self.__call__(options)

    def inspect_dataset(self, scan_meshlab, colmap_model, occlusions=None, splats=None, image_path=None):
        if image_path is None:
            image_path = self.image_path
        options = ["DatasetInspector", "--scan_alignment_path", scan_meshlab,
                   "--image_base_path", image_path, "--state_path", colmap_model,
                   "--max_occlusion_depth", str(self.max_occlusion_depth)]
        if occlusions is not None:
            options += ["--occlusion_mesh_path", occlusions]
        if splats is not None:
            options += ["--occlusion_splats_path", splats]
        self.__call__(options)
