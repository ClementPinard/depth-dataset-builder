from colmap_util import database as db
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import pandas as pd
import numpy as np
from sqlite3 import IntegrityError
from tqdm import tqdm
parser = ArgumentParser(description='Add video frames to a colmap db file. '
                        'Needs original colmap db file to link the frames to the right cameras',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--frame_list', metavar='PATH',
                    help='path to list with relative path to images', type=Path, default=None)
parser.add_argument('--metadata', metavar='PATH', required=True,
                    help='path to metadata csv file', type=Path)
parser.add_argument('--database', metavar='DB', required=True,
                    help='path to colmap database file, to get the image ids right')


def add_to_db(db_path, metadata, frame_list_path, **env):
    database = db.COLMAPDatabase.connect(db_path)

    frame_list = []
    frame_ids = []
    if frame_list_path is not None:
        with open(frame_list_path, "r") as f:
            frame_list = [line[:-1] for line in f.readlines()]
        metadata = metadata[metadata["image_path"].isin(frame_list)]

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image_path = row["image_path"]
        camera_id = row["camera_id"]
        if "location_valid" in row.keys() and row["location_valid"]:
            frame_gps = row[["location_longitude", "location_latitude", "location_altitude"]]
        else:
            frame_gps = np.full(3, np.NaN)
        try:
            frame_ids.append(database.add_image(image_path, int(camera_id), prior_t=frame_gps, image_id=row["db_id"]))
        except IntegrityError:
            sql_string = "SELECT camera_id, image_id FROM images WHERE name='{}'".format(image_path)
            sql_output = next(database.execute(sql_string))
            existing_camera_id = sql_output[0]
            assert(existing_camera_id == camera_id)
            frame_ids.append(sql_output[1])
    database.commit()
    database.close()
    return frame_ids


def get_frame_without_features(db_path):
    database = db.COLMAPDatabase.connect(db_path)
    first_string = "SELECT image_id FROM descriptors WHERE cols=0"
    descriptors = list(database.execute(first_string))
    for d in descriptors:
        second_string = "SELECT name FROM images WHERE image_id={}".format(d)
        row = list(database.execute(second_string))[0]
        print(row)

    database.close()


def main():
    args = parser.parse_args()
    add_to_db(args.database, pd.read_csv(args.metadata), args.frame_list)

    return


if __name__ == '__main__':
    main()
