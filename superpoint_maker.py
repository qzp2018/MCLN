# superpoints fetch
import segmentator
import open3d as o3d
import torch
import os
from six.moves import cPickle
import numpy as np

# BRIEF read from pkl
def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data()."""
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


def generate_superpoint(data_path, data_path_scannet, split):


    scans = unpickle_data(f'{data_path}/{split}_v3scans.pkl')
    scans = list(scans)[0]

    for scan in scans:
        spformer_file = os.path.join(data_path_scannet, split, scan + "_vh_clean_2.ply")
        mesh = o3d.io.read_triangle_mesh(spformer_file)
        vertices = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
        faces = torch.tensor(np.array(mesh.triangles), dtype=torch.int64)
        superpoint = segmentator.segment_mesh(vertices, faces)
        select_idx = torch.tensor(scans[scan].choices)
        superpoint = torch.index_select(superpoint, 0, select_idx).numpy()
        torch.save(superpoint, os.path.join(data_path, "superpoints", split, scan + "_superpoint.pth"))
        print("Saving " + scan)

    print("Done.")


if __name__ == '__main__':
    data_path = r"/path/to/scanrefer"  # ScanRefer path
    data_path_scannet = r"/path/to/scannetv2"  # ScanNetv2 path
    split = 'train'
    generate_superpoint(data_path=data_path, data_path_scannet=data_path_scannet, split=split)



