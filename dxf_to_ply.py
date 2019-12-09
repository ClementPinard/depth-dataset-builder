import numpy as np
import ezdxf
import meshio
from path import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm


def dxf2numpy(dxf_file, centroid):
    print("Opening dxf file...")
    file = ezdxf.readfile(dxf_file)
    msp = file.modelspace()
    print("extracting edges from dxf file...")
    num_edges = sum(1 for _ in msp.query('POLYLINE'))
    edges = np.zeros((num_edges, 2, 3))

    for i, pl in tqdm(enumerate(msp.query('POLYLINE')), total=num_edges):
        start, end = pl.points()
        edges[i, 0] = start
        edges[i, 1] = end
    edges -= centroid
    return edges.astype(np.float32)


def edges2triangles(edges):
    vertices, edge_indices = np.unique(edges.reshape(-1, 3), axis=0, return_inverse=True)
    edge_indices = edge_indices.reshape(-1, 2)

    vertex_tree = {}

    def add_entry(tree, loc, target):
        if loc not in tree:
            tree[loc] = set()
        tree[loc].add(target)

    print("Constructing vertex tree...")
    for seg in tqdm(edge_indices):
        i_start, i_end = seg
        add_entry(vertex_tree, i_start, i_end)
        add_entry(vertex_tree, i_end, i_start)

    faces_set = set()

    print("Detecting triangles...")

    def sub_dict(tree, indices):
        return {k: tree[k] for k in indices if k in tree}

    for v1, leaf1 in tqdm(vertex_tree.items()):
        for v2, leaf2 in sub_dict(vertex_tree, leaf1).items():
            for v3, leaf3 in sub_dict(vertex_tree, leaf2).items():
                if v1 in leaf3:
                    faces_set.add(frozenset([v1, v2, v3]))

    faces = np.zeros((len(faces_set), 3), dtype=np.int32)
    for i, f in enumerate(faces_set):
        faces[i] = list(f)

    return vertices, faces


parser = ArgumentParser(description='convert a dxf file with only edges to a faced mesh, only counting triangles',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--dxf', default="manoir.dxf",
                    help='dxf file, must contain the wireframe')
parser.add_argument('--centroid_path', default="centroid.txt",
                    help='txt containing the centroid computed with las2ply.py')
parser.add_argument('--output', default=None,
                    help="output file name. By default, will be dxf path with \".dxf\" replace with \"ply\"")


def main():
    args = parser.parse_args()
    if args.centroid_path is not None:
        centroid = np.loadtxt(args.centroid_path)
    else:
        centroid = np.zeros(3)

    if args.output is None:
        output_name = Path(args.dxf).stripext()
        output_path = str(output_name + ".ply")
    else:
        output_path = args.output

    edges = dxf2numpy(args.dxf, centroid)
    vertices, faces = edges2triangles(edges)

    cells = {
        "triangle": faces
    }
    meshio.write_points_cells(
        output_path,
        vertices,
        cells,
        file_format='ply-ascii'
    )


if __name__ == '__main__':
    main()
