import numpy as np
import pandas as pd
import sklearn.manifold
import umap
import kmapper as km
from gtda.diagrams import PersistenceLandscape
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import TakensEmbedding
from kmapper import Cover
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn import ensemble
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

def load_and_prep_data():
    path = './data'

    # testing_set = pd.read_csv(f'{path}/UNSW_NB15_testing-set.csv')
    # training_set = pd.read_csv(f'{path}/UNSW_NB15_training-set.csv')
    # LIST_EVENTS = pd.read_csv(f'{path}/UNSW-NB15_LIST_EVENTS.csv')
    # GT = pd.read_csv(f'{path}/NUSW-NB15_GT.csv')

    # NB15_1 = pd.read_csv(f'{path}/UNSW-NB15_1.csv')
    # NB15_2 = pd.read_csv(f'{path}/UNSW-NB15_2.csv')
    NB15_3 = pd.read_csv(f'{path}/UNSW-NB15_3.csv')
    NB15_4 = pd.read_csv(f'{path}/UNSW-NB15_4.csv')
    NB15_features = pd.read_csv(f'{path}/NUSW-NB15_features.csv', encoding='cp1252')

    # NB15_1.columns = NB15_features['Name']
    # NB15_2.columns = NB15_features['Name']
    NB15_3.columns = NB15_features['Name']
    NB15_4.columns = NB15_features['Name']

    df = pd.concat([
        # NB15_1,
        # NB15_2,
        NB15_3,
        NB15_4
    ], ignore_index=False)

    uniques = df['attack_cat'].unique()

    df['attack_cat'] = df['attack_cat'].replace(np.nan, 0)
    df['attack_cat'] = df['attack_cat'].replace('Exploits', 11)
    df['attack_cat'] = df['attack_cat'].replace(' Fuzzers ', 12)
    df['attack_cat'] = df['attack_cat'].replace('DoS', 13)
    df['attack_cat'] = df['attack_cat'].replace('Generic', 14)
    df['attack_cat'] = df['attack_cat'].replace(' Shellcode ', 15)
    df['attack_cat'] = df['attack_cat'].replace('Backdoor', 16)
    df['attack_cat'] = df['attack_cat'].replace(' Reconnaissance ', 17)
    df['attack_cat'] = df['attack_cat'].replace('Worms', 18)
    df['attack_cat'] = df['attack_cat'].replace('Analysis', 19)
    df['attack_cat'] = pd.to_numeric(df['attack_cat'])

    df['Timestamp'] = pd.to_datetime(df['Stime'], unit='s')
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)

    return df

df = load_and_prep_data()

### Plotting Raw Data
a_df = df[df['Label'] != 0]
b_df = df[df['Label'] == 0]
# atts = ['dur', 'sbytes', 'dbytes']
# fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
# for i, col in enumerate(atts):
#     axs[i].plot(b_df.index, b_df[col], color='green', linewidth=1, alpha=0.5, label='Benign')
#     axs[i].plot(a_df.index, a_df[col], color='red', linewidth=1, label='Attack')
#     axs[i].set_title(f'{col}: Benign vs Attack')
#     axs[i].set_ylabel(col)
#     axs[i].legend()
# plt.xlabel('Time')
# plt.tight_layout()
# plt.show()

uni = df['attack_cat'].unique()

### Plotting Aggregated Data
agg_df = df.resample('1s').agg(
    dur=('dur', 'sum'),
    sbytes=('sbytes', 'sum'),
    dbytes=('dbytes', 'sum'),
    sttl=('sttl', 'sum'),
    dttl=('dttl', 'sum'),
    sloss=('sloss', 'sum'),
    dloss=('dloss', 'sum'),
    Sload=('Sload', 'sum'),
    Dload=('Dload', 'sum'),
    Spkts=('Spkts', 'sum'),
    Dpkts=('Dpkts', 'sum'),
    Sjit=('Sjit', 'sum'),
    Djit=('Djit', 'sum'),
    # attack_cat=('attack_cat', lambda x: stats.mode(x)[0]),
    # label=('Label', lambda x: stats.mode(x)[0]),
    attack_cat=('attack_cat', lambda x: stats.mode(x)[0]),
    label=('Label', 'max'),
)
agg_df.fillna(0, inplace=True)
a_agg = agg_df[agg_df['label'] == 1]
b_agg = agg_df[agg_df['label'] == 0]
atts = ['sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
    'Sload', 'Dload', 'Spkts', 'Dpkts', 'Sjit', 'Djit']
# fig, axes = plt.subplots(3, 4, figsize=(12, 12))
# for i, attr in enumerate(atts):
#     row, col = divmod(i, 4)
#     axes[row, col].plot(b_agg.index, b_agg[attr], color='blue')
#     axes[row, col].plot(a_agg.index, a_agg[attr], color='red')
#     axes[row, col].set_title(attr)
# plt.tight_layout()
# plt.show()

### K-distance elbow plot
scaler = StandardScaler()
df_scaled = scaler.fit_transform(agg_df[atts])

# neighbors = NearestNeighbors()
# neighbors_fit = neighbors.fit(df_scaled)
# distances, indices = neighbors_fit.kneighbors(df_scaled)
# distances = np.sort(distances, axis=0)
# distances = distances[:, 1]
#
# plt.plot(distances)
# plt.xlabel('distance index')
# plt.ylabel('average-distance')
# plt.title('K-distance elbow plot')
# plt.show()

### Testing different Mapper lenses
mapper = km.KeplerMapper(verbose=1)
lens1 = mapper.fit_transform(df_scaled, projection='l2norm')

projector = ensemble.IsolationForest(random_state=42)
projector.fit(df_scaled)
lens2 = projector.decision_function(df_scaled)

pca = PCA(n_components=1)
lens3 = pca.fit_transform(df_scaled)

# fig, axs = plt.subplots(1, 2, figsize=(9,4))
# axs[0].scatter(lens1,lens2,alpha=0.3)
# axs[0].set_xlabel('L^2-Norm')
# axs[0].set_ylabel('IsolationForest')
# axs[1].scatter(lens1,lens3,alpha=0.3)
# axs[1].set_xlabel('L^2-Norm')
# axs[1].set_ylabel('PCA')
# plt.tight_layout()
# plt.show()

## Mapper
uniques = agg_df['attack_cat'].unique()
lens = np.c_[lens1, lens2]
cover = Cover(n_cubes=20, perc_overlap=0.20)
clusterer = DBSCAN(eps=0.5, min_samples=5)

G = mapper.map(
    lens,
    df_scaled,
    cover=cover,
    clusterer=clusterer,
)

_ = mapper.visualize(
    G,
    custom_tooltips=agg_df['attack_cat'].values,
    color_values=agg_df['attack_cat'].values,
    color_function_name="Label",
    path_html="mapper_unsw-nb15_agg.html",
    X=df_scaled,
    X_names=agg_df.columns,
    lens=lens,
)

### Takens with separate benigns and attacks
# dim = len(atts)
#
# b_sliced = b_agg[atts].iloc[:int(b_agg.index.size/10)]
# a_sliced = a_agg[atts].iloc[:int(a_agg.index.size/10)]
#
# b_scaled = scaler.fit_transform(b_sliced)
# b_scaled = pd.DataFrame(b_scaled, index=b_sliced.index, columns=atts)
# a_scaled = scaler.fit_transform(a_sliced)
# a_scaled = pd.DataFrame(a_scaled, index=a_sliced.index, columns=atts)
#
# te = TakensEmbedding(time_delay=1, dimension=3)
# b_embeddings = te.fit_transform(b_scaled)
# a_embeddings = te.fit_transform(a_scaled)
#
# ph = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
# b_diagrams = ph.fit_transform(b_embeddings)
# a_diagrams = ph.fit_transform(a_embeddings)
#
# markers = {0: 'x', 1: 'o', 2: '^'}
# labels = {0: 'H0', 1: 'H1', 2: 'H2'}
# max_death = 0
#
# fig = plt.subplots(figsize=(6, 6))
# for d in [0,1,2]:
#     dgm = b_diagrams[0][b_diagrams[0][:, 2] == d]
#     if len(dgm) > 0:
#         plt.scatter(dgm[:, 0], dgm[:, 1], c='green', marker=markers[d],
#                     label=f'{labels[d]} Benign', alpha=0.5)
#         max_death = max(max_death, max(dgm[:, 1]))
#
# for d in [0,1,2]:
#     dgm = a_diagrams[0][a_diagrams[0][:, 2] == d]
#     if len(dgm) > 0:
#         plt.scatter(dgm[:, 0], dgm[:, 1], c='red', marker=markers[d],
#                     label=f'{labels[d]} Attack', alpha=0.5)
#         max_death = max(max_death, max(dgm[:, 1]))
#
# plt.plot([0, max_death], [0, max_death], 'k--', alpha=0.5)
# plt.xlabel('Birth')
# plt.ylabel('Death')
# plt.title('Persistence Diagrams (H0, H1, H2) takens')
# plt.legend()
# plt.grid(True)
# plt.show()

### Persistence landscapes
# pl = PersistenceLandscape()
# b_ls = pl.fit_transform(b_diagrams)
# a_ls = pl.fit_transform(a_diagrams)

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# for i in range(b_ls.shape[1]):
#     ax[0].plot(b_ls[0, i], label=f'Benign')
#     ax[0].set_title('Persistence Landscapes (H0, H1, H2) benign')
# for i in range(a_ls.shape[1]):
#     ax[1].plot(a_ls[0, i], label=f'Attack')
#     ax[1].set_title('Persistence Landscapes (H0, H1, H2) attack')
# plt.show()


### Euler Characteristic Profile
# from eulearning.utils import vectorize_st, codensity
# from eulearning.descriptors import EulerCharacteristicProfile
# import gudhi as gd
#
# b_vec_sts = []
# for i in range(b_embeddings.shape[0]):
#     b_point_cloud = b_embeddings[i]
#     b_ac = gd.AlphaComplex(points=b_point_cloud)
#     b_st = b_ac.create_simplex_tree()
#     X_ = np.array([b_ac.get_point(i) for i in range(b_st.num_vertices())])
#     b_codensity_filt = codensity(X_)
#     b_vec_st = vectorize_st(b_st, filtrations=[b_codensity_filt])
#     b_vec_sts.append(b_vec_st)
#
# a_vec_sts = []
# for i in range(a_embeddings.shape[0]):
#     a_point_cloud = a_embeddings[i]
#     a_ac = gd.AlphaComplex(points=a_point_cloud)
#     a_st = a_ac.create_simplex_tree()
#     Y_ = np.array([a_ac.get_point(i) for i in range(a_st.num_vertices())])
#     a_codensity_filt = codensity(Y_)
#     a_vec_st = vectorize_st(a_st, filtrations=[a_codensity_filt])
#     a_vec_sts.append(a_vec_st)
#
# euler_profile = EulerCharacteristicProfile(resolution=(50,50), quantiles=[(0, 1), (0, 1)], pt_cld=True, normalize=False, flatten=True)
# b_ecp = euler_profile.fit_transform(b_vec_sts)
# a_ecp = euler_profile.fit_transform(a_vec_sts)
# extent = list(euler_profile.val_ranges[0])+list(euler_profile.val_ranges[1])
#
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(b_ecp, origin='lower', extent=extent, aspect='auto')
# ax[0].set_title('Euler characteristic surface - Benign')
# ax[1].imshow(a_ecp, origin='lower', extent=extent, aspect='auto')
# ax[1].set_title('Euler characteristic surface - Attack')
# plt.show()