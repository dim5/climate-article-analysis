# %%
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Tuple
tqdm.pandas()

df = pd.read_csv('../../data/dump_cleaned.csv.gz', index_col='Id')
#%%
bannedSites = (
    "kcrg.com", "Peninsula Daily News", "State of Delaware News", "KOMU.com",
    "www.plenglish.com/", "www.plenglish.com", "Fortune", 'techcrunch.com',
    'finance.yahoo.com', 'sports.yahoo.com', 'news.yahoo.com', 'adage.com',
    'Fiji Broadcasting Corporation,'
    'businessgreen.com', 'Axios', 'huffpost.com', 'TribLIVE.com', 'EW.com',
    'news-reporter.com', 'Nature', 'coleofduty.com', 'prnewswire.com',
    'prnewswire.com'
    'alexareports.com', 'Science', 'GlobeNewswire News Room',
    'businesswire.com', 'Bandera County Courier', 'gcu.org', 'ArchDaily',
    'SFChronicle.com', 'Cornell Chronicle', 'openPR.com', 'East Bay Express',
    'IPWatchdog.com | Patents & Patent Law', 'Trends Wide'
)  # bad parsing: paywall, gdpr, copyright, spam OR science journals

df = df[~df.SiteName.isin(bannedSites)]
df = df[~df.Url.str.contains('.edu', regex=False)]
df = df.sort_values('Created').drop_duplicates('Content', 'first')

#%% remove bad articles
df = df.drop(df[((df.SiteName == 'Time') &
                 (df.Content.str.startswith('Welcome!'))) |
                ((df.SiteName == 'Common Dreams') &
                 (df.Content.str.startswith('Dear Common Dreams'))
                 | df.Content.str.startswith('Welcome! Meredith collects data')
                 | df.Content.str.contains('^.{1,30}is part of Verizon Media.')
                 | df.Content.str.startswith('Extended Forecast\n\n'))].index)

#%% remove bad sections
from itertools import takewhile
import re

end_exp = re.compile("""^ *\*\[.+$""")
pollmsg = 'If you can\'t see this reader poll, please refresh your page.'


def backinfo_remove(x: str):
    split = x.split('\n')
    if split[0].startswith("Media playback is unsupported on your device") \
      or split[0].startswith("Image copyright"):
        split = split[1:]

    endlen = len(
        list(takewhile(lambda y: end_exp.match(y) or y == pollmsg,
                       split[::-1])))
    if endlen > 0:
        return '\n'.join(split[:-endlen - 1])
    return '\n'.join(split)


df.Content = df.Content.apply(backinfo_remove)

#%%
df_sorted = df.sort_values('Created')
dedupe_content = ~df_sorted.Content.duplicated('first')
dedupe_title = ~df_sorted.Title.duplicated('first')
dedupe_title |= df.Title.isin(('BBVA VENEZUELA', 'Letters to the Editor',
                               'Middle-east Arab News Opinion'))

df = df[dedupe_content & dedupe_title]

#%% pd-dedupe
import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_lg")


def spacy_counts(articles: List[str]) -> List[Tuple[int, str, int]]:
    counts = []
    for doc in tqdm(nlp.pipe(articles,
                             batch_size=200,
                             disable=['tagger', 'ner', 'parser']),
                    total=len(articles)):
        lemmas = [
            t.lemma_.lower() for t in doc if not (t.is_punct or t.is_stop)
        ]
        lemmatized = " ".join(lemmas)
        word_count = sum(1 for token in doc if not token.is_punct)
        counts.append((word_count, lemmatized, len(lemmas)))
    return counts


new_cols = pd.Series(spacy_counts(df.Content), index=df.index)

df['WordCount'], df['StrippedContent'], df['LemmaCount'] = zip(*new_cols)

#%%df
df_sorted = df.sort_values('Created')
df = df[~df_sorted.StrippedContent.duplicated('first')]

#%%
df['lemmaratio'] = df.apply(lambda x: 1 - x['LemmaCount'] / x['WordCount'], axis=1)

wmean = df.WordCount.mean()
wstd = df.WordCount.std()

lmean = df.lemmaratio.mean()
lstd = df.lemmaratio.std()

df_word_red = df[(df.WordCount >= wmean - wstd)
                 & (df.WordCount <= wmean + 2 * wstd)]

print(len(df_word_red) / len(df))

df_l_red = df_word_red[(df.lemmaratio >= lmean - 2 * lstd)]
df_l_red.drop(['lemmaratio'], axis=1, inplace=True)
df_l_red.to_csv('../../data/processed.csv.gz')