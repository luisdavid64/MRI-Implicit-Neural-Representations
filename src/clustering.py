import argparse
from models.utils import get_config, get_data_loader 
import fastmri
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans 
import numpy as np
from math import sqrt
import fastmri
from collections import OrderedDict

def partition_kspace(dataset = None, img=None, kcoords=None, show = True, no_steps=40, no_parts=4):
    if dataset == None and (img == None or kcoords == None):
        raise ValueError('Dataset or image must be provided')
    if dataset:
        C,H,W,S = dataset.shape
        img = dataset.image.reshape(C,H,W,S)
        kcoords = dataset.coords.reshape(C,H,W,3)
    C,H,W,S = img.shape 

    dist_to_center = torch.sqrt(kcoords[...,1]**2 + kcoords[...,2]**2)
    # Max distance when |x|=1 and |y|=1, so sqrt(2)
    inds = []
    for i in range(no_steps):
        if i == 0:
            r_0 = 0
        else:
            r_0 = sqrt(2)*(i)/no_steps
        if i == no_steps - 1:
            r_1 = sqrt(2)
        else:
            r_1 = sqrt(2)*(i + 1)/no_steps
        ind = torch.where((dist_to_center >= r_0) & (dist_to_center <= r_1))
        inds.append(ind)
    radial_data = [torch.log(fastmri.complex_abs(img[ind])) for ind in inds]
    means = [torch.mean(part).item() for part in radial_data]
    means = np.array(means).reshape(-1,1)
    kmeans = KMeans(
        init="random",
        n_clusters=no_parts,
        n_init=10,
        max_iter=200,
        random_state=42
    )
    kmeans.fit(means)
    # With radii going outwards, so garantied to have order
    labels = kmeans.labels_
    unique_elements, indices, counts = np.unique(labels, return_counts=True, return_index=True)
    # Relative order of clusters is important
    order = np.argsort(indices)
    unique_elements = unique_elements[order]
    counts = counts[order]
    normalized_counts = sqrt(2)*np.cumsum(counts / len(labels))

    # Add center as starting point
    radii = [0] + list(OrderedDict(zip(unique_elements, normalized_counts)).values())
    radii = np.array(radii)

    # Make sure last one covers entire range
    radii[no_parts] = 2
    # We can ignore the Coil as not relevant
    if show:
        clustered = np.zeros((C,H,W))
        for (ind,label) in zip(inds,labels):
            clustered[ind] = label
        plt.imshow(clustered[0], cmap='gray')
        plt.show()
    return labels, radii

def partition_and_stats(dataset = None, img=None, kcoords=None, show = True, no_steps=40, no_parts=4, stat="max"):
    if dataset == None and (img == None or kcoords == None):
        raise ValueError('Dataset or image must be provided')
    if dataset:
        C,H,W,S = dataset.shape
        img = dataset.image.reshape(C,H,W,S)
        kcoords = dataset.coords.reshape(C,H,W,3)
    _, radii = partition_kspace(dataset,img,kcoords, show, no_steps, no_parts)
    dist_to_center = torch.sqrt(kcoords[...,1]**2 + kcoords[...,2]**2)
    stats = []
    for i in range(len(radii) - 1):
        r_0 = radii[i]
        r_1 = radii[i+1]
        ind = torch.where((dist_to_center >= r_0) & (dist_to_center <= r_1))
        if stat == "min":
            st = torch.abs(img[ind]).min()
            stats.append(st)
        else:
            st = torch.abs(img[ind]).max()
            stats.append(st)
    return stats, radii
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config/local/config_siren_kspace_loe.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")

    # Load experiment setting
    opts = parser.parse_args()
    config = get_config(opts.config)
    # The only difference of val loader is that data is not shuffled
    dataset, _, _ = get_data_loader(
        data=config['data'], 
        data_root=config['data_root'], 
        set=config['set'], 
        batch_size=config['batch_size'],
        transform=config['transform'], 
        num_workers=0, 
        sample=config["sample"], 
        slice=config["slice"],
        shuffle=True,
        full_norm=config["full_norm"],
        normalization=config["normalization"]
    )
    C,H,W,S = dataset.shape
    img = dataset.image.reshape(C,H,W,S)
    coords = dataset.coords.reshape(C,H,W,3)
    partition_and_stats(img=img,kcoords=coords, show=True)