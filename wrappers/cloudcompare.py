from .default_wrapper import Wrapper


class CloudCompare(Wrapper):
    def __init__(self, binary):
        super().__init__(binary)
        self.base_options = ["-SILENT", "-AUTO_SAVE", "OFF", "-C_EXPORT_FMT", "PLY"]

    def compute_normals_mst(self, output_file, input_file, mst_knn=6):
        options = ["-O", input_file, "-OCTREE_NORMALS", "auto",
                   "-ORIENT_NORMS_MST", str(mst_knn), "-SAVE_CLOUDS_FILE", output_file]
        self.__call__(self.base_options + options)
