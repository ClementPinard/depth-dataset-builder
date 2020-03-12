from colmap_util import database as db
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import pandas as pd
import numpy as np
from sqlite3 import IntegrityError
from tqdm import tqdm
parser = ArgumentParser(description='Create vizualisation for specified video',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--frame_list', metavar='PATH',
                    help='path to list with relative path to images', type=Path, default=None)
parser.add_argument('--metadata', metavar='PATH',
                    help='path to metadata csv file', type=Path)
parser.add_argument('--database', metavar='DB', required=True,
                    help='path to colmap database file, to get the image ids right')


def add_to_db(db_path, metadata_path, frame_list_path, **env):
    metadata = pd.read_csv(metadata_path)
    database = db.COLMAPDatabase.connect(db_path)

    frame_list = []
    if frame_list_path is not None:
        with open(frame_list_path, "r") as f:
            frame_list = [line[:1] for line in f.readlines()]
        metadata = metadata[metadata["image_path"].isin(frame_list)]

    for image_id, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image_path = row["image_path"]
        camera_id = row["camera_id"]
        if row["location_valid"]:
            frame_gps = row[["location_longitude", "location_latitude", "location_altitude"]]
        else:
            frame_gps = np.full(3, np.NaN)
        try:
            database.add_image(image_path, int(camera_id), prior_t=frame_gps)
        except IntegrityError:
            sql_string = "SELECT camera_id FROM images WHERE name='{}'".format(image_path)
            existing_camera_id = print(next(database.execute(sql_string))[0])
            assert(existing_camera_id == camera_id)


def main():
    args = parser.parse_args()
    add_to_db(args.database, args.metadata, args.frame_list)

    return


if __name__ == '__main__':
    main()
