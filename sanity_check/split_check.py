import os, h5py, hashlib, numpy as np

def img_hashes(h5_path):
    with h5py.File(h5_path, 'r') as f:
        imgs = f['images'][:]
    # hash each image consistently
    return {hashlib.md5(np.ascontiguousarray(im).view(np.uint8)).hexdigest() for im in imgs}

out_dir = "C:/Users/Intel User/Desktop/blackbird_scripts/data/hdf5_files/cross_validation/two_class"
sets = [img_hashes(os.path.join(out_dir, f"test_{i}.hdf5")) for i in range(5)]
for i in range(5):
    for j in range(i+1, 5):
        print(f"overlap test_{i} vs test_{j}: {len(sets[i] & sets[j])}")
