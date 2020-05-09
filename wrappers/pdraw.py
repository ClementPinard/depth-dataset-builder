from .default_wrapper import Wrapper
import tempfile
import pandas as pd


class PDraw(Wrapper):
    def vmeta_extract(self, video):
        temp = tempfile.NamedTemporaryFile()
        options = ["vmeta-extract", str(video), "--csv", temp.name]
        self.__call__(options)
        return pd.read_csv(temp.name, sep=" ")
