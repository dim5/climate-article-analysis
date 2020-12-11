#%%
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

df = pd.read_csv('../../data/database_dump.csv', index_col='Id')
df.head()

# %%
import html2text
h = html2text.HTML2Text()
h.ignore_links = True
h.ignore_images = True
h.ignore_tables = True
h.ignore_emphasis = True

df['Content'] = df['Content'].progress_apply(lambda x: (h.handle(x)).strip())
df['Len'] = df['Content'].progress_apply(lambda x: len(x))


#%%
def url_to_site(url: str):
    from urllib.parse import urlparse
    return urlparse(url).netloc.lstrip('www.')


df['SiteName'] = df.apply(
    lambda row: row.SiteName
    if not pd.isnull(row.SiteName) else url_to_site(row.Url).lstrip('www.'),
    axis=1)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

#%%
df.dropna(inplace=True)
df.to_csv('../data/dump_cleaned.csv.gz')
