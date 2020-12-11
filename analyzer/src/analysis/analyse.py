#%%
from datetime import date
from typing import DefaultDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_colwidth', 999)
#%%
df = pd.read_csv('../../data/clustered.csv.gz', index_col='Id')
df.Created = df.Created.apply(pd.to_datetime)
# %%


def show_samples(clustering: str, count=5, every_n=1, store=df, other_cols=[]):
    for i in range(0, store[clustering].max().astype(int).item() + 1, every_n):
        display(store[store[clustering] == i].sample(
            n=count,
            random_state=42)[['Title', 'SiteName', clustering, *other_cols]])


# %%
just_clustered = df.dropna()
# %%
clusterings = [
    'TfIdfClusters', 'lda_clust', 'bert_avg_clust', 'bert_max_clust'
]
titles = ['LSA', 'LDA', 'BERT avg', 'BERT max']
#%%
for clust, label in zip(clusterings, titles):
    sns.kdeplot(df[clust], label=label)

plt.xlabel("Cluster ID")
plt.xticks(range(0, 12 + 1))
plt.legend()
plt.show()

#%%
from itertools import chain
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axit = chain.from_iterable(zip(*axes))
for clust, color, ax, title in zip(clusterings, sns.color_palette("tab10"),
                                   axit, titles):
    sns.histplot(
        df[clust],
        ax=ax,
        bins=np.arange(int(df[clust].max() + 2)) - 0.5,  #center tick
        color=color,
        kde='True')
    ax.set_xlabel("Cluster ID")
    ax.set_xticks(range(0, int(df[clust].max()) + 1))
    ax.set_yticks(range(0, 3501, 500))
    ax.set_title(title)
plt.tight_layout(pad=0.4, w_pad=1, h_pad=1.0)
plt.show()


# %%
def show_value(ax):
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height() / 2
        value = '{:.2f}'.format(p.get_height())
        ax.text(_x, _y, value, ha="center")


ax = sns.barplot(x=titles,
                 y=[len(df[i].dropna()) / len(df) * 100 for i in clusterings])
plt.ylabel("Clustered (%)")
plt.xlabel("Embedding technique")
show_value(ax)

#%%%%% groupings
import pickle
from collections import Counter
ner_cnts = {}
with open('filtered_dict.pt', 'rb') as f:
    ner_cnts = pickle.load(f)

date_ner_counts = {
    i: dict(Counter([df.loc[ix].Created.date() for ix, _ in ner_cnts[i]]))
    for i in ner_cnts
}

date_df = pd.DataFrame.from_dict(date_ner_counts)
date_df.index = pd.to_datetime(date_df.index)
date_df.sort_index(inplace=True)
date_df.fillna(0, inplace=True)
#%%
date_df = date_df['2019-12-01':]  #actively collecting from this date

# %%
weekly = date_df[:'2020-09-05'].resample('W-MON').sum()
monthly = date_df[:'2020-08-31'].resample('MS').sum()


# %%
def show_time(ner: str,
              color=(0.12156862745098039, 0.4666666666666667,
                     0.7058823529411765),
              store=date_df,
              label='day'):
    sns.lineplot(x=store.index, y=ner.upper(), data=store, color=color)
    plt.xticks(rotation=25)
    plt.ylabel(f'Article count per {label}')
    plt.xlabel('Date')
    plt.show()


def show_weekly(ner: str,
                color=(0.12156862745098039, 0.4666666666666667,
                       0.7058823529411765)):
    show_time(ner, color, weekly, 'week')


def show_monthly(ner: str,
                 color=(0.12156862745098039, 0.4666666666666667,
                        0.7058823529411765)):
    show_time(ner, color, monthly, 'month')


# %%
for i in [
        'DONALD TRUMP', 'COVID-19', 'PARIS AGREEMENT', 'GRETA THUNBERG',
        'Australia'
]:
    print(i)
    show_weekly(i)
# %%
places = [
    'EARTH', 'UNITED STATES', 'AUSTRALIA', 'CHINA', 'AMERICA', 'EUROPE',
    'UNITED KINGDOM', 'CALIFORNIA'
]
monthly.drop(places, axis=1).idxmax(axis=1)
# %%
