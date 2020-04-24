import numpy as np
from lxml import etree
from path import Path


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


def add_mesh_to_project(input_mlp, output_mlp, model_path, index=0, label=None, transform=np.eye(4)):
    with open(input_mlp, "r") as f:
        to_modify = etree.parse(f)
    if label is None:
        label = model_path.basename()
    root = to_modify.getroot()
    group = root[0]
    mesh = etree.Element("MLMesh")
    mesh.set("label", label)
    mesh.set("filename", model_path)
    matrix = etree.SubElement(mesh, "MLMatrix44")
    matrix.text = "\n" + "\n".join(" ".join(str(element) for element in row) + " " for row in transform) + "\n"
    group.insert(index, mesh)
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


if __name__ == '__main__':
    model_paths = [Path("path/to/model.ply"), Path("path/to/other.ply")]
    labels = "1", "2"
    transforms = [np.random.randn(4, 4), np.random.randn(4, 4)]
    create_project("test.mlp", model_paths)
    add_mesh_to_project("test.mlp", "test.mlp", model_paths[0], index=0)

    matrix, filename = remove_mesh_from_project("test.mlp", "test2.mlp", 0)
    print(matrix, filename)
