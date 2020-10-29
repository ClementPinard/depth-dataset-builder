import numpy as np
from lxml import etree
from path import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def create_project(mlp_path, model_paths, labels=None, transforms=None):
    if labels is not None:
        assert(len(model_paths) == len(labels))
    else:
        labels = [m.basename() for m in model_paths]

    if transforms is not None:
        assert(len(model_paths) == len(transforms))
    else:
        transforms = [np.eye(4) for _ in model_paths]
    base = etree.Element("MeshLabProject")
    group = etree.SubElement(base, "MeshGroup")
    for m, l, t in zip(model_paths, labels, transforms):
        mesh = etree.SubElement(group, "MLMesh")
        mesh.set("label", l)
        mesh.set("filename", m)
        matrix = etree.SubElement(mesh, "MLMatrix44")
        matrix.text = "\n" + "\n".join(" ".join(str(element) for element in row) + " " for row in t) + "\n"
    tree = etree.ElementTree(base)
    tree.write(mlp_path, pretty_print=True)


def remove_mesh_from_project(input_mlp, output_mlp, index):
    with open(input_mlp, "r") as f:
        to_modify = etree.parse(f)
    meshgroup = to_modify.getroot()[0]
    if index < len(meshgroup):
        removed = meshgroup[index]
        meshgroup.remove(removed)
    to_modify.write(output_mlp, pretty_print=True)
    transform = np.fromstring(removed[0].text, sep=" ").reshape(4, 4)
    filepath = removed.get("label")
    return transform, filepath


def get_mesh(input_mlp, index):
    with open(input_mlp, "r") as f:
        to_modify = etree.parse(f)
    meshgroup = to_modify.getroot()[0]
    if index < len(meshgroup):
        removed = meshgroup[index]
    transform = np.fromstring(removed[0].text, sep=" ").reshape(4, 4)
    filepath = removed.get("label")
    return transform, filepath


def add_meshes_to_project(input_mlp, output_mlp, model_paths, labels=None, transforms=None, start_index=-1):
    if labels is not None:
        assert(len(model_paths) == len(labels))
    else:
        labels = [m.basename() for m in model_paths]

    if transforms is not None:
        assert(len(model_paths) == len(transforms))
    else:
        transforms = [np.eye(4) for _ in model_paths]
    with open(input_mlp, "r") as f:
        to_modify = etree.parse(f)
    root = to_modify.getroot()
    group = root[0]
    if start_index < 0:
        start_index = len(group)
    for i, (m, l, t) in enumerate(zip(model_paths, labels, transforms)):
        mesh = etree.Element("MLMesh")
        mesh.set("label", l)
        mesh.set("filename", m)
        matrix = etree.SubElement(mesh, "MLMatrix44")
        matrix.text = "\n" + "\n".join(" ".join(str(element) for element in row) + " " for row in t) + "\n"
        group.insert(start_index, mesh)
    to_modify.write(output_mlp, pretty_print=True)


def apply_transform_to_project(input_mlp, output_mlp, transform):
    with open(input_mlp, "r") as f:
        to_modify = etree.parse(f)
    meshgroup = to_modify.getroot()[0]
    for mesh in meshgroup:
        former_transform = np.fromstring(mesh[0].text, sep=" ").reshape(4, 4)
        new_transform = transform @ former_transform
        mesh[0].text = "\n" + "\n".join(" ".join(str(element) for element in row) + " " for row in new_transform) + "\n"
    to_modify.write(output_mlp, pretty_print=True)


parser = ArgumentParser(description='Create a meshlab project with ply files and transformations',
                        formatter_class=ArgumentDefaultsHelpFormatter)

subparsers = parser.add_subparsers(dest="operation")
create_parser = subparsers.add_parser('create')
create_parser.add_argument('--input_models', metavar='PLY', type=Path, nargs="+")
create_parser.add_argument('--output_meshlab', metavar='MLP', type=Path, required=True)
create_parser.add_argument('--transforms', metavar='TXT', type=Path, nargs="+")
create_parser.add_argument('--labels', metavar='LABEL', nargs="*")

remove_parser = subparsers.add_parser('remove')
remove_parser.add_argument('--input_meshlab', metavar='MLP', type=Path, required=True)
remove_parser.add_argument('--output_meshlab', metavar='MLP', type=Path, required=True)
remove_parser.add_argument('--index', metavar="N", type=int, default=-1)

add_parser = subparsers.add_parser('add')
add_parser.add_argument('--input_models', metavar='PLY', type=Path, nargs="+")
add_parser.add_argument('--input_meshlab', metavar='MLP', type=Path, required=True)
add_parser.add_argument('--output_meshlab', metavar='MLP', type=Path, required=True)
add_parser.add_argument('--transforms', metavar='TXT', type=Path, nargs="+")
add_parser.add_argument('--labels', metavar='LABEL', nargs="*")
add_parser.add_argument('--start_index', metavar='N', default=-1, type=int)

transform_parser = subparsers.add_parser('transform')
transform_parser.add_argument('--input_meshlab', metavar='MLP', type=Path, required=True)
transform_parser.add_argument('--output_meshlab', metavar='MLP', type=Path, required=True)
transform_parser.add_argument('--transform', metavar='TXT', type=Path, required=True)
transform_parser.add_argument('--inverse', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.operation in ["add", "create"]:
        n_models = len(args.input_models)
        if args.labels is not None:
            assert n_models == len(args.labels)

        if args.transforms is None:
            transforms = [np.eye(4, 4)] * n_models
        elif len(args.transforms) == 1:
            transform = np.fromfile(args.transforms[0], sep=" ").reshape(4, 4)
            transforms = [transform] * n_models
        else:
            assert n_models == len(transforms)
            transforms = [np.fromfile(t, sep=" ").reshape(4, 4) for t in args.transforms]
        if args.operation == "create":
            create_project(args.output_meshlab, args.input_models, args.labels, transforms)
        if args.operation == "add":
            add_meshes_to_project(args.input_meshlab,
                                  args.output_meshlab,
                                  args.input_models,
                                  args.labels,
                                  transforms,
                                  args.start_index)
    if args.operation == "remove":
        matrix, filename = remove_mesh_from_project(args.input_meshlab, args.output_meshlab, args.index)
        print("Removed model {} with transform\n {} \nfrom meshlab".format(filename, matrix))
    if args.operation == "transform":
        transform = np.fromfile(args.transform, sep=" ").reshape(4, 4)
        if args.inverse:
            transform = np.linalg.inverse(transform)
        apply_transform_to_project(args.input_meshlab, args.output_meshlab, transform)
