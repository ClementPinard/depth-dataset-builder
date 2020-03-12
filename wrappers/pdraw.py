from .default_wrapper import Wrapper
import tempfile
import pandas as pd


class PDraw(Wrapper):
    def __init__(self, wrapper_file, quiet=False):
        super().__init__(wrapper_file, quiet)

    def vmeta_extract(self, video):
        temp = tempfile.NamedTemporaryFile()
        options = ["vmeta-extract", str(video), "--csv", temp.name]
        self.__call__(options)
        return pd.read_csv(temp.name, sep=" ")
