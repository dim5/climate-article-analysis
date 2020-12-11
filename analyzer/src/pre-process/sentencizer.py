import pandas as pd
from tqdm.auto import tqdm
from typing import List
tqdm.pandas()
# %%
df = pd.read_csv('../../data/processed.csv.gz', index_col='Id')
#%%
import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_lg")


def spacy_sentencer(articles: List[str]) -> List[List[str]]:
    sentencized = []
    for doc in tqdm(nlp.pipe(articles,
                             batch_size=200,
                             disable=['tagger', 'ner']),
                    total=len(articles)):
        sents = [sent.text.strip() for sent in doc.sents if len(sent) > 1]
        sentencized.append(sents)
    return sentencized


sents = pd.Series(spacy_sentencer(df.Content), index=df.index)

sents.to_pickle('../../data/processed_sents.pt')