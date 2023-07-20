import h5py
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict 


def generate_cluster(vid, k, feats):
    n, d = feats[vid]['c3d_features'].shape
    sampled_idx = np.linspace(0, n-1, num=129)
    sampled_idx = np.round((sampled_idx[0:-1] + sampled_idx[1:]) / 2).astype(int)

    # print(sampled_idx)
    s_feats = np.array(feats[vid]['c3d_features'])[sampled_idx]
    s_feats = s_feats / np.linalg.norm(s_feats, axis=-1, keepdims=True)
    sims = s_feats @ s_feats.T
    x = np.concatenate([sims, np.sqrt(np.arange(128)[:,None])], axis=1)
    
    kmeans = KMeans(n_clusters=k).fit(x)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    cluster_dis = ((cluster_centers[None, :, :] - cluster_centers[:, None, :]) ** 2).sum(axis=-1)

    cluster_size = [(labels == cluster_id).sum() for cluster_id in range(k)]
    for cluster_id in range(k):
        mask = (labels == cluster_id)
        n_cluster = mask.sum()
        if n_cluster < 11:
            neighbors = np.argsort(cluster_dis[cluster_id], axis=0)
            for neighbor in neighbors:
                if cluster_size[neighbor] >= 11:
                    labels[mask] = neighbor
                    break

    clusters = []
    for cluster_id in range(n):
        mask = (labels == cluster_id)
        if (mask.sum()):
            assert mask.sum() >= 11
            clusters.append([np.arange(128)[mask].min(), np.arange(128)[mask].max()])
    
    return sorted(clusters, key=lambda x:x[0])


if __name__=='__main__':
    feats = h5py.File('data/activitynet/sub_activitynet_v1-3.c3d.hdf5')
    events = defaultdict(list)
    for vid in tqdm(list(feats.keys())):
        atomic = generate_cluster(vid, 5, feats)
        for i in range(len(atomic)):
            for j in range(i, len(atomic)):
                events[vid].append([min(atomic[i][0], atomic[j][0]), max(atomic[i][1], atomic[j][1])])

    with open('events.pkl', 'wb') as f:
        pickle.dump(events, f)
