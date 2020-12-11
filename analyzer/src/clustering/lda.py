from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score, silhouette_samples
import matplotlib.cm as cm


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " | ".join(
            [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# %%
lda = LDA(n_components=10,
          verbose=1,
          learning_method='batch',
          random_state=42,
          max_iter=1000,
          evaluate_every=5,
          n_jobs=-1,
          max_doc_update_iter=500)

tf_df = tfidf.div(idfs, axis=1)  # transform back to df
lda.fit(tf_df)
print_top_words(lda, tfidf.columns, 5)

# %% clustering based on LDA
X = lda.transform(tf_df.values)
km = MiniBatchKMeans(n_clusters=10, random_state=42)
km.fit(X)
print(silhouette_score(X, km.labels_, sample_size=1000))
print(km.inertia_)
