# -*- coding: utf-8 -*-

# import hotspot
import sys

import libpysal
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.cluster import spectral_clustering
import pandas as pd
from scipy.sparse import issparse
from mudata import MuData
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import decomposition
import scanpy as sc
import numpy as np
# from scMVP.dataset import GeneExpressionDataset, CellMeasurement
from scipy.spatial import distance
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import os
import math
import scipy.stats as st
import muon as mu
from muon import atac as ac
from sklearn.mixture import GaussianMixture as GMM
# import community as community_louvain
from matplotlib import RcParams
from scipy import stats
from sklearn import metrics
from spreg import ML_Lag

np.random.seed(0)
import warnings

warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from collections import defaultdict
import geopandas as gpd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import pairwise_distances
from libpysal.weights import DistanceBand
from esda.moran import Moran, Moran_Local
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import linkage, leaves_list
import numpy as np


def readData():
    print(os.getcwd())
    rna_file = "./data/Mouse_Thymus/adata_RNA.h5ad"
    rna = sc.read_h5ad(rna_file)
    dataset = "Mouse_Thymus"
    atac_file = "./data/Mouse_Thymus/adata_ADT.h5ad"
    atac = sc.read_h5ad(atac_file)
    return rna, atac, dataset


def pre_rna(adata):
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat_v3", )
    return adata


def pre_atac(adata):
    # sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=200)
    return adata


def spatial_smoothing(expression_matrix, coords, k=5):
    n_spots, n_genes = expression_matrix.shape

    # 计算最近邻
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # 初始化插补后的矩阵
    smoothed_matrix = np.zeros_like(expression_matrix)

    for i in range(n_spots):
        neighbors = indices[i]
        smoothed_matrix[i] = np.mean(expression_matrix[neighbors], axis=0)

    return smoothed_matrix


def compute_similarity_matrix(gene_expression_data):
    gene_names = gene_expression_data.var_names
    # 计算基因之间的余弦相似性矩阵
    similarity_matrix = cosine_similarity(gene_expression_data.X.T)

    # 将相似性矩阵转换为DataFrame，并设置基因名称作为行和列的索引
    similarity_matrix_df = pd.DataFrame(similarity_matrix, index=gene_names, columns=gene_names)
    print("similarity_matrix_df:", similarity_matrix_df.shape)
    return similarity_matrix_df


def hierarchical_clustering(similarity_matrix, num_modules):
    """
    Perform hierarchical clustering based on the cosine similarity matrix.
    Args:
        similarity_matrix (DataFrame): The similarity matrix of genes.
        num_modules (int): The number of clusters/modules to form.
    Returns:
        modules (array): Cluster assignments for each gene.
        linkage_matrix (array): Linkage matrix for hierarchical clustering.
    """
    # 将相似度矩阵转化为距离矩阵
    cosine_distance_matrix = 1 - similarity_matrix.to_numpy()
    np.fill_diagonal(cosine_distance_matrix, 0)

    # 打印距离矩阵以便检查
    cosine_distance_matrix_df = pd.DataFrame(cosine_distance_matrix,
                                             index=similarity_matrix.index,
                                             columns=similarity_matrix.columns)
    # print(cosine_distance_matrix_df)

    # 将距离矩阵转换为方阵格式
    # distance_matrix = squareform(cosine_distance_matrix)
    distance_matrix = squareform(cosine_distance_matrix_df)
    print(distance_matrix.shape)
    # # 处理奇异矩阵：添加一个小常数以避免奇异矩阵
    # epsilon = 1e-6  # 可以根据需要调整这个值
    # np.fill_diagonal(distance_matrix, np.diag(distance_matrix) + epsilon)

    # 层次聚类
    try:
        # 使用不同的聚类方法，避免“ward”方法可能带来的问题
        linkage_matrix = linkage(distance_matrix, method='ward')  # 你可以尝试其他方法：'single', 'complete', 'average'

        # 根据链接矩阵分配基因到不同的模块
        modules = fcluster(linkage_matrix, num_modules, criterion='maxclust')

        return modules, linkage_matrix
    except Exception as e:
        print(f"聚类失败: {e}")
        return None, None


def get_knn(adata, label, n_components, k_num=50):
    knn = {k: [] for k in ["group_" + str(i) for i in range(n_components)]}
    for i in range(n_components):
        print("label[i]:", len(label[i]))
        if label[i]:
            # Compute KNN of cells for a specific feature group
            knn_dist = kneighbors_graph(adata.X[:, label[i]], mode="connectivity", n_neighbors=k_num)
            knn_dist_coo = knn_dist.tocoo()
            neighbors_dict = defaultdict(list)
            for j, k in zip(knn_dist_coo.row, knn_dist_coo.col):
                neighbors_dict[j].append(k)

            for j in neighbors_dict:
                knn["group_" + str(i)].extend(neighbors_dict[j][:k_num])
    return knn


from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster

from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix


def compute_similarity_matrix_with_spatial(adata, spatial_key, spatial_weight=0.5, k=10):
    expression_data = adata.X
    if hasattr(expression_data, "toarray"):  # 如果是稀疏矩阵
        expression_data = expression_data.toarray()

    spatial_coords = adata.obsm[spatial_key]

    expression_similarity = cosine_similarity(expression_data.T)

    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)
    spatial_similarity = lil_matrix((adata.n_obs, adata.n_obs))
    for i, neighbors in enumerate(indices):
        for j, dist in zip(neighbors, distances[i]):
            spatial_similarity[i, j] = np.exp(-dist ** 2 / (2 * 10 ** 2))

    weighted_expression_data = spatial_similarity @ expression_data
    weighted_expression_similarity = cosine_similarity(weighted_expression_data.T)

    combined_similarity = spatial_weight * weighted_expression_similarity + (1 - spatial_weight) * expression_similarity

    return combined_similarity


from sklearn.metrics.pairwise import pairwise_kernels

from sklearn.cluster import SpectralClustering

from sklearn.cluster import AgglomerativeClustering

from collections import Counter


def cluster_and_balance_with_spatial_consistency(
        adata, spatial_coords_key,
        n_clusters=10, max_features_per_cluster=2000, min_features_per_cluster=2,
        spatial_weight=0.6, spatial_consistency_weight=0.6
):
    # 确保我们对基因（列）进行聚类
    feature_matrix = adata.X.T  # 转置特征矩阵
    current_spatial_coords = adata.obsm[spatial_coords_key]

    # 计算基因间相似性矩阵
    similarity_matrix = compute_similarity_matrix_with_spatial(adata, spatial_coords_key)

    # 使用谱聚类代替层次聚类
    spectral_clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',  # 因为我们已经有了相似性矩阵
        assign_labels='discretize',
        random_state=42  # 设置随机种子以保证结果可复现
    )

    # 直接使用相似性矩阵作为输入
    initial_labels = spectral_clustering.fit_predict(similarity_matrix)
    print("Initial labels:", initial_labels)

    # 平衡类别分布并考虑空间一致性
    balanced_labels = balance_clusters_with_spatial_consistency(
        similarity_matrix,
        initial_labels,
        max_features_per_cluster,
        min_features_per_cluster,
        spatial_weight,
        spatial_consistency_weight
    )

    return balanced_labels


import numpy as np
from collections import Counter


def balance_clusters_with_spatial_consistency(
        similarity_matrix, initial_labels, max_features_per_cluster, min_features_per_cluster,
        spatial_weight=0.6, spatial_consistency_weight=0.6, max_iterations=100
):
    adjusted_labels = initial_labels.copy()
    n_clusters = len(np.unique(initial_labels))
    unique_labels = np.arange(n_clusters)

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        label_counts = Counter(adjusted_labels)

        # 找出过小类别
        small_clusters = [c for c, count in label_counts.items() if count < min_features_per_cluster]
        if not small_clusters:
            break

        # 从最大类别中分配特征到过小类别
        largest_cluster = max(label_counts, key=label_counts.get)
        largest_cluster_indices = np.where(adjusted_labels == largest_cluster)[0]

        for small_cluster in small_clusters:
            if len(largest_cluster_indices) > min_features_per_cluster:
                for idx in largest_cluster_indices[:min_features_per_cluster]:
                    # 动态调整类别
                    target_cluster = np.argmax([
                        spatial_weight * similarity_matrix[idx, adjusted_labels == c].mean() +
                        spatial_consistency_weight * (max_features_per_cluster - label_counts[c])
                        if c != largest_cluster else -np.inf
                        for c in unique_labels
                    ])
                    adjusted_labels[idx] = target_cluster

            # 更新最大类别索引
            largest_cluster_indices = np.where(adjusted_labels == largest_cluster)[0]

    return adjusted_labels


def get_knn_corr(knn, n_components):
    knn_corr = {k: 0 for k in ["group_" + str(i) for i in range(n_components)]}
    tmp = []
    # The similarity between feature groups was calculated using jaccard
    for i in range(n_components):
        for j in range(n_components):
            tmp.append(1 - distance.jaccard(knn["rna"]["group_" + str(i)],
                                            knn["atac"]["group_" + str(j)]))
        max_k = 0
        flag = 0
        # Find the most similar feature groups
        for k in range(n_components):
            if tmp[k] > max_k and k not in knn_corr.values():
                max_k = tmp[k]
                flag = k
        knn_corr["group_" + str(i)] = flag
        tmp = []
    return knn_corr


from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jaccard


def spatial_distance(group_cells, cell_coords):
    # group_coords = cell_coords.iloc[group_cells].to_numpy()  # 如果是位置索引则使用 iloc
    group_coords = cell_coords[group_cells]  # 提取特征组中细胞的坐标
    return np.mean(group_coords, axis=0)  # 计算该特征组的空间中心（加权均值）


def get_knn_corr_global(knn, n_components, cell_coords, spatial_weight=0.5):
    cost_matrix = np.zeros((n_components, n_components))

    for i in range(n_components):
        rna_cells = knn["rna"]["group_" + str(i)]
        rna_spatial_center = spatial_distance(rna_cells, cell_coords)

        for j in range(n_components):
            atac_cells = knn["atac"]["group_" + str(j)]
            atac_spatial_center = spatial_distance(atac_cells, cell_coords)

            jaccard_sim = 1 - jaccard(rna_cells, atac_cells)
            spatial_dist = np.linalg.norm(rna_spatial_center - atac_spatial_center)

            cost_matrix[i, j] = -(
                    spatial_weight * jaccard_sim +
                    (1 - spatial_weight) * spatial_dist
            )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    knn_corr = {f"group_{i}": {"atac": f"{col_ind[i]}"} for i in range(n_components)}
    return knn_corr


def integrate(mdata, label, knn_corr, n_components, mofa_factor=2, latent_dim=10):
    var_names = []
    rna_name = "rna"
    atac_name = "atac"
    for i in range(n_components):
        if label[rna_name][i] and label[atac_name][i]:
            var_name = mu.MuData({
                rna_name: mdata[rna_name][:, label[rna_name][i]],
                atac_name: mdata[atac_name][:, label[atac_name][int(knn_corr["group_" + str(i)]["atac"])]]})
            if var_name[i]:
                # Use mofa to integrate feature groups between different omics
                mu.tl.mofa(var_name, n_factors=mofa_factor)
            var_names.append(var_name)
    # Save the mofa integration results
    integrate_dim = []
    for i in range(n_components):
        if label[rna_name][i] and label[atac_name][i]:
            for j in range(mofa_factor):
                integrate_dim.append(var_names[i].obsm["X_mofa"][:, j])
    integrate_array = np.asarray(integrate_dim).T
    # Dimensionality reduction using pca
    return sc.tl.pca(integrate_array, n_comps=latent_dim)
    # return integrate_array


if __name__ == "__main__":

    rna, atac, dataname = readData()
    # print("################")
    print(dataname)
    rna.var_names_make_unique()
    atac.var_names_make_unique()
    celltype = {"sciCAR_cellline": "labels", "snare_p0": "cell_type", "snare_cellline": "cell_line",
                "share_skin": "celltype", "10x_pbmc": "label", "kidney": "cell_name", "neuips": "cell_type",
                "10x_lymph_node": "cell_type", "snare_AdBrainCortex": "cell_type"}

    rna.var_names_make_unique()
    atac.var_names_make_unique()
    rna_final = pre_rna(rna)
    atac_final = pre_atac(atac)

    feature_rna = rna_final[:, rna_final.var["highly_variable"]]
    feature_atac = atac_final[:, atac_final.var["highly_variable"]]
    print(feature_atac.X.shape[1])

    n_cluster_rna = 10
    n_cluster_atac = 10
    k_num = 50

    n_factor = 2

    gene_clusters = cluster_and_balance_with_spatial_consistency(feature_rna, "spatial", n_cluster_rna)
    atac_clusters = cluster_and_balance_with_spatial_consistency(feature_atac, "spatial", n_cluster_rna)
    print(gene_clusters)

    label_pre_rna = dict([(k, []) for k in range(n_cluster_rna)])
    for i in range(n_cluster_rna):
        for j in range(feature_rna.X.shape[1]):
            if gene_clusters[j] == i:
                label_pre_rna[i].append(j)

    label_pre_atac = dict([(k, []) for k in range(n_cluster_rna)])
    for i in range(n_cluster_rna):
        for j in range(feature_atac.X.shape[1]):
            if atac_clusters[j] == i:
                label_pre_atac[i].append(j)

    from collections import Counter

    label_counts = Counter(gene_clusters)
    # 输出每个簇中的基因数量
    for cluster_id, count in sorted(label_counts.items()):
        print(f"Cluster {cluster_id}: {count} genes")

    label_counts = Counter(atac_clusters)
    # 输出每个簇中的基因数量
    for cluster_id, count in sorted(label_counts.items()):
        print(f"Cluster {cluster_id}: {count} pro")

        # 设置图形参数
    sc.set_figure_params(figsize=(5, 4), dpi_save=300)
    plt.rcParams['font.size'] = 10
    mods_name = ["rna", "atac"]
    mdata_pre = mu.MuData({"rna": feature_rna, "atac": feature_atac})

    label = {}
    label["rna"] = label_pre_rna
    label["atac"] = label_pre_atac
    knn = {}
    for mod in mods_name:
        knn[mod] = get_knn(mdata_pre[mod], label[mod], n_cluster_rna, k_num)

    for mod in mods_name:
        for j in range(n_cluster_rna):
            print(len(knn[mod]["group_" + str(j)]))

    knn_corr = get_knn_corr_global(knn, n_cluster_rna, feature_rna.obsm["spatial"])
    # knn_corr = get_knn_corr(knn,n_cluster_rna)
    print("knn_corr:", knn_corr)
    obsm_name = "X_mofa"
    feature_rna.obsm[obsm_name] = integrate(mdata_pre, label, knn_corr, n_cluster_rna)

    spatial_df = pd.DataFrame(feature_rna.obsm["spatial"])
    print("Original columns:", spatial_df.columns)

    feature_rna.obsm["spatial"] = spatial_df.to_numpy()

    # 计算邻居图
    sc.pp.neighbors(feature_rna, use_rep=obsm_name, key_added="neighbors_mofa")

    # Leiden聚类
    sc.tl.leiden(feature_rna, neighbors_key="neighbors_mofa", key_added="leiden_mofa", resolution=0.3)

    # 计算UMAP
    sc.tl.umap(feature_rna, neighbors_key="neighbors_mofa")

    # 绘制UMAP图并保存
    sc.pl.umap(feature_rna, color='leiden_mofa', title='Spa_scMFG', s=50, show=False,
               save=dataname + "_Spa_scMFG_average_knncorr_global_spectral_clustering_no_smoth_" + str(
                   k_num) + "_" + str(n_cluster_rna) + "_" + str(n_cluster_atac) + "_" + str(n_factor) + ".png")

    # 绘制空间图并保存
    sc.pl.embedding(feature_rna, basis='spatial', color='leiden_mofa', title='Spa_scMFG', s=50, show=False,
                    save=dataname + "_Spa_scMFG_average_knncorr_global_spectral_clustering_no_smoth_" + str(
                        k_num) + "_" + str(n_cluster_rna) + "_" + str(n_cluster_atac) + "_" + str(
                        n_factor) + "_spatial.png")

    # 调整布局并显示
    filename = dataname + "_" + str(
        n_cluster_rna) + "_average_knncorr_global_spectral_clustering_seurat_v3_3000_" + str(k_num) + "_" + str(
        n_cluster_atac) + "_" + str(n_factor) + ".h5ad"
    feature_rna.write_h5ad(filename)
