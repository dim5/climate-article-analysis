import pickle
import spacy
from typing import List, Tuple, Dict, Union
import pandas as pd
from collections import defaultdict, Counter
from tqdm.auto import tqdm

spacy.prefer_gpu()

df = pd.read_csv('../../data/processed.csv.gz', index_col="Id")

filtered = {}
with open('../data/filtered_dict.pt', 'rb') as fil:
    filtered = pickle.load(fil)

filtered_kept_articles = {
    item[0]
    for _, sublist in filtered.items() for item in sublist
}
print


#%%
def calc_tf(this_ner_count: int, all_ner_count: int) -> float:
    return this_ner_count / all_ner_count


def calc_idf(all_doc_cnt: int, docs_with_this_ner: int) -> float:
    return math.log(all_doc_cnt / docs_with_this_ner)


'''
TF(t) = (Number of times term t appears in a document) /
        (Total number of terms in the document)
        
IDF(t) = log_e(Total number of documents /
         Number of documents with term t in it)

tf-idf := tf * idf
'''

tfidf = pd.DataFrame(index=filtered_kept_articles)
for i in filtered:
    ent = pd.DataFrame(filtered[i], columns=['art', 'cnt'])
    ent.set_index('art', inplace=True)
    tfidf[i] = ent

# %%
# tfidf is a DataFrame with named entities as columns and
# each cell contains the count of that entity in that article
total_doc_cnt = len(filtered_kept_articles)

idfs = tfidf.apply(lambda docs: calc_idf(total_doc_cnt, docs.count()), axis=0)
tf_ner_sum = tfidf.apply(lambda x: x.sum(), axis=1)

tfidf = tfidf.div(tf_ner_sum, axis=0)
tfidf = tfidf.mul(idfs, axis=1)

tfidf.fillna(0, inplace=True)

# %% -------------------------- Hugging embed ----
#-------------------------------------------------

import torch
import numpy as np


def make_embeddings():
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0]  #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def create_embedding():
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/roberta-large-nli-stsb-mean-tokens",
            use_fast=True)
        model = AutoModel.from_pretrained(
            "sentence-transformers/roberta-large-nli-stsb-mean-tokens")

        model.eval()

        def embed_art(text: List[str], avgpool=True):
            it = iter(text)
            text = [t + next(it, '') + next(it, '') + next(it, '') for t in it]

            with torch.no_grad():
                encoded_input = tokenizer(text,
                                          padding=True,
                                          truncation=True,
                                          max_length=512,
                                          return_tensors='pt')
                model_output = model(**encoded_input)
                sentence_embeddings = mean_pooling(
                    model_output, encoded_input['attention_mask'])
            pool = F.adaptive_avg_pool2d if avgpool else F.adaptive_max_pool2d
            return pool(sentence_embeddings.unsqueeze(0),
                        (1, 1024)).squeeze(0).squeeze(0).cpu().data.numpy()

        sents: Series = pd.read_pickle('../../data/processed_sents.pt')
        sents = sents[sents.index.isin(filtered_kept_articles)]
        return sents.progress_apply(lambda x: embed_art(x))

    return create_embedding()


# %% ----------clustering -----------
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
import umap
import hdbscan
import seaborn as sns

df_filtered = df[df.index.isin(filtered_kept_articles)].copy()


def display_clustering(clustered_data, cluster_labels, hide_axis_labels=False):
    if clustered_data.shape[1] != 2:
        umap_data = umap.UMAP(n_neighbors=30,
                              n_components=2,
                              min_dist=0.0,
                              random_state=42).fit_transform(clustered_data)
    else:
        umap_data = clustered_data
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = cluster_labels

    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result[result.labels == -1]
    clustered = result[result.labels != -1]

    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x,
                clustered.y,
                c=clustered.labels,
                s=0.1,
                cmap='Spectral')
    if hide_axis_labels:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    cbar = plt.colorbar(ticks=range(0, np.max(cluster_labels) + 1))
    cbar.set_label('Cluster ID')
    plt.show()


#%% --------- tfidf clust
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD

MIN_CLUSTER_SIZE = round(total_doc_cnt / 100 / 2)


def tfidf_clust(clusterer, lsa=False, target_dim=100):
    def do_lsa(X, target_dim):
        svd = TruncatedSVD(target_dim, random_state=42)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        return lsa.fit_transform(X)

    X = tfidf.values if not lsa else do_lsa(tfidf.values, target_dim)
    res = clusterer.fit_predict(X)
    print(silhouette_score(X, clusterer.labels_))
    # print(km.inertia_)
    return res


hdb = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                      metric='euclidean',
                      cluster_selection_method='eom',
                      prediction_data=False,
                      cluster_selection_epsilon=0.38,
                      min_samples=MIN_CLUSTER_SIZE * 2)
res = tfidf_clust(hdb, True, 16)
print(hdb.labels_.max())
#display_clustering(tfidf.values, res)

#%%
tfidf['TfIdfClusters'] = res
df_filtered['TfIdfClusters'] = tfidf['TfIdfClusters']
del tfidf['TfIdfClusters']


# %% ------------------- LDA
def lda_clust():
    from sklearn.decomposition import LatentDirichletAllocation as LDA

    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " | ".join([
                feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]
            ])
            print(message)
        print()

    lda = LDA(n_components=10,
              verbose=1,
              learning_method='batch',
              random_state=42,
              max_iter=1000,
              evaluate_every=5,
              n_jobs=-1,
              max_doc_update_iter=500)

    tf_df = tfidf.div(idfs, axis=1).copy()  # transform back to df
    X = lda.fit_transform(tf_df)
    print_top_words(lda, tfidf.columns, 5)
    # %% clustering based on LDA
    km = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                         metric='euclidean',
                         cluster_selection_method='leaf',
                         prediction_data=False,
                         cluster_selection_epsilon=0.01,
                         min_samples=MIN_CLUSTER_SIZE)
    ret = km.fit_predict(X)
    print(silhouette_score(X, km.labels_, sample_size=1000))
    print(km.labels_.max())

    tf_df['LDA Clusters'] = ret

    # print(km.inertia_)
    return X, ret, tf_df['LDA Clusters']


lda_data, lda_clusts, ldaclustseries = lda_clust()
with open('../../data/clustering/ldaout.pt', 'wb') as f:
    pickle.dump((lda_data, lda_clusts), f, pickle.HIGHEST_PROTOCOL)

df_filtered['lda_clust'] = ldaclustseries
# %% ---------------bert clust
text_embed_avg = pd.read_pickle('../../data/text_embed_pooled_avg.pt')
text_embed_max = pd.read_pickle('../../data/text_embed_pooled_max.pt')

#%%
text_embed_avg = text_embed_avg[text_embed_avg.index.isin(
    filtered_kept_articles)]
text_embed_max = text_embed_max[text_embed_max.index.isin(
    filtered_kept_articles)]


# %%
def cluster_bert(data):
    stacked = np.stack(data)
    X = umap.UMAP(n_neighbors=20,
                  min_dist=0.0,
                  n_components=30,
                  random_state=42).fit_transform(stacked)
    hdb_bert = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                               metric='euclidean',
                               cluster_selection_method='leaf',
                               cluster_selection_epsilon=0.4,
                               min_samples=MIN_CLUSTER_SIZE)
    res = hdb_bert.fit_predict(X)
    sh = silhouette_score(X, hdb_bert.labels_, sample_size=1000)
    lmax = hdb_bert.labels_.max()
    print(f"sh: {sh}\t lmax: {lmax}")
    # display_clustering(X, res, True)

    return pd.Series(res, index=data.index)


#%%

df_filtered = pd.concat(
    (df_filtered, cluster_bert(text_embed_avg).rename("bert_avg_clust")),
    axis=1)
df_filtered = pd.concat(
    (df_filtered, cluster_bert(text_embed_max).rename("bert_max_clust")),
    axis=1)

# %%
df_filtered = df_filtered.replace(-1, np.nan)

#%%
df_filtered.to_csv('../../data/clustered.csv.gz', compression="gzip")
