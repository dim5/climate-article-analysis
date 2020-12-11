#%%
from typing import List, Tuple, Dict, Union
import pandas as pd

df = pd.read_csv('../../data/processed.csv.gz', index_col="Id")
pd.set_option('display.max_colwidth', 999)

# %% ************ getting ners
import spacy
#%%
import pickle
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_lg")

banned_ner = {('FAC', 'FAHRENHEIT'), ('GPE', 'NEW'),
              ('ORG', 'CLIMATE HOME NEWS'),
              ('ORG', 'E&E NEWS'), ('ORG', 'E&E'), ('ORG', 'AFP'),
              ('ORG', 'GETTY IMAGES'), ('ORG', 'GETTY'), ('ORG', 'GUARDIAN'),
              ('ORG', 'ASSOCIATED PRESS'), ('ORG', 'AP'), ('ORG', 'BLOOMBERG'),
              ('ORG', 'CBS NEWS'), ('ORG', 'CNN'), ('ORG', 'FOX NEWS'),
              ('ORG', 'NATURE COMMUNICATIONS'), ('ORG', 'NATURE'),
              ('ORG', 'NEW YORK TIMES'), ('ORG', 'REUTERS'),
              ('ORG', 'THE ASSOCIATED PRESS'), ('ORG', 'THE CANADIAN PRESS'),
              ('ORG', 'THE NEW YORK TIMES'), ('ORG', 'THE WASHINGTON POST'),
              ('ORG', 'THE'), ('ORG', 'THOMSON REUTERS FOUNDATION'),
              ('ORG', 'THOMSON REUTERS'), ('ORG', 'TIME'), ('ORG', 'TIMES'),
              ('PERSON', 'FAHRENHEIT'), ('PERSON', 'DAVID'),
              ('PERSON', 'EUREKALERT'), ('ORG', 'EUREKALERT'),
              ('PERSON', 'PH.D'), ('WORK_OF_ART', 'NATURE CLIMATE CHANGE'),
              ('WORK_OF_ART', 'NATURE'), ('WORK_OF_ART', 'PHD'),
              ('WORK_OF_ART', 'SCIENCE'), ('WORK_OF_ART', 'THE CONVERSATION')}


def get_ner_pairs(df: pd.DataFrame):
    label_filter = {
        'CARDINAL', 'ORDINAL', 'QUANTITY', 'MONEY', 'PERCENT', 'TIME', 'DATE'
    }
    for i, doc in enumerate(
            nlp.pipe(df.Content, batch_size=200, disable=['tagger',
                                                          'parser'])):
        site = df.SiteName.iloc[i]
        pairs = [
            pair for ent in doc.ents if ent.label_.upper() not in label_filter
            and site.upper() not in ent.lemma_.upper() and (pair := (
                ent.label_.upper(), ent.lemma_.upper())) not in banned_ner
        ]
        yield pairs


#%% *****---- filtering them
from collections import defaultdict, Counter


def count_merge(pairs) -> Dict[Tuple, int]:
    d = defaultdict(int)
    for label, text in pairs:
        text = text.strip()
        text = text[4:] if text[:4] == "THE " else text
        text = text[:-2] if text[-2:] in ('’S', "'S") else text
        text = " ".join(text.split())  # removes double spaces
        if len(text) > 1:
            d[(label, text)] += 1
    return dict(d)


from tqdm.auto import tqdm
tqdm.pandas()
docu_ner = {}

for i, pairs in tqdm(enumerate(get_ner_pairs(df)), total=len(df)):
    counts = count_merge(pairs)
    docu_ner[df.index.values[i]] = counts

#%% ****** inverted index
inv_ner = defaultdict(list)

for docu_id, counts in tqdm(docu_ner.items()):
    for (label, text), count in counts.items():
        inv_ner[(label, text)].append((docu_id, count))
inv_ner = dict(inv_ner)


#%% ****** NER merging: business rules
def ner_merger(merge_to: Union[Tuple, str],
               *merge_from: Union[Tuple, str],
               store=inv_ner) -> None:
    if not merge_to in store:
        store[merge_to] = []

    c = Counter(dict(store[merge_to]))
    for mergee in merge_from:
        if merge_to == mergee:
            continue
        elif not mergee in store:
            print(f"{mergee} not in dict")
        else:
            c += Counter(dict(store[mergee]))
            del store[mergee]
    store[merge_to] = list((c).items())


#%%%
ner_merger(('GPE', 'UNITED KINGDOM'), ('GPE', 'THE UNITED KINGDOM'),
           ('GPE', 'U.K'), ('GPE', 'UK'), ('GPE', 'U.K.'))
ner_merger(('GPE', 'UNITED STATES'), ('GPE', 'THE UNITED STATES'),
           ('GPE', 'THE UNITED STATES OF AMERICA'), ('GPE', 'U.S'),
           ('GPE', 'U.S.'), ('GPE', 'U.S.A.'), ('GPE', 'US'), ('GPE', 'USA'))
ner_merger(('GPE', 'UNITED ARAB EMIRATES'),
           ('GPE', 'THE UNITED ARAB EMIRATES'), ('GPE', 'U.A.E.'),
           ('GPE', 'UAE'))

ner_merger(('LOC', 'INDIAN OCEAN'), ('LOC', 'THE INDIAN OCEAN'))
ner_merger(('LOC', 'MIDDLE EAST'), ('LOC', 'THE MIDDLE EAST'))
ner_merger(('LOC', 'PACIFIC OCEAN'), ('LOC', 'PACIFIC'),
           ('LOC', 'THE PACIFIC OCEAN'))

ner_merger(('ORG', 'BRITISH PETROL'), ('ORG', 'BP'))
ner_merger(('ORG', 'CENTER FOR DISEASE CONTROL'), ('ORG', 'CDC'))
ner_merger(('ORG', 'EXXON'), ('ORG', 'EXXONMOBIL'))
ner_merger(('ORG', 'HARVARD UNIVERSITY'), ('ORG', 'HARVARD'))
ner_merger(('ORG', 'STANFORD UNIVERSITY'), ('ORG', 'STANFORD'))
ner_merger(('ORG', 'EU'), ('ORG', 'EUROPEAN UNION'), ('ORG', 'E.U.'))
ner_merger(('ORG', 'EPA'), ('ORG', 'THE ENVIRONMENTAL PROTECTION AGENCY'))

ner_merger(('PERSON', 'BERNIE SANDERS'), ('PERSON', 'BERNIE'),
           ('PERSON', 'SANDERS'))
ner_merger(('PERSON', 'JOE BIDEN'), ('PERSON', 'BIDEN'))
ner_merger(('PERSON', 'GRETA THUNBERG'), ('PERSON', 'GRETA'),
           ('PERSON', 'THUNBERG'))
ner_merger(('PERSON', 'DONALD TRUMP'), ('PERSON', 'DONALD'),
           ('PERSON', 'TRUMP'), ('PERSON', 'PRESIDENT DONALD TRUMP'),
           ('PERSON', 'PRESIDENT TRUMP'))
ner_merger(('PERSON', 'SCOTT MORRISON'), ('PERSON', 'SCOTT'),
           ('PERSON', 'MORRISON'))
ner_merger(('PERSON', 'OBAMA'), ('PERSON', 'BARACK OBAMA'),
           ('PERSON', 'BARACK'), ('PERSON', 'PRESIDENT OBAMA'))

ner_merger(('PERSON', 'NANCY PELOSI'), ('PERSON', 'PELOSI'))
ner_merger(('PERSON', 'JUSTIN TRUDEAU'), ('PERSON', 'TRUDEAU'))
ner_merger(('PERSON', 'JEFF BEZOS'), ('PERSON', 'BEZOS'))
ner_merger(('PERSON', 'GEORGE FLOYD'), ('PERSON', 'FLOYD'))
ner_merger(('PERSON', 'GEORGE BUSH'), ('PERSON', 'BUSH'))
ner_merger(('PERSON', 'ANGELA MERKEL'), ('PERSON', 'MERKEL'))
ner_merger(('PERSON', 'BILL GATES'), ('PERSON', 'GATES'))
ner_merger(('PERSON', 'JAIR BOLSONARO'), ('PERSON', 'BOLSONARO'))
ner_merger(('PERSON', 'WARREN BUFFET'), ('PERSON', 'BUFFET'))

ner_merger(('ORG', 'INTERNATIONAL ENERGY AGENCY'), ('ORG', 'IEA'))
ner_merger(('GPE', 'WASHINGTON, D.C.'), ('GPE', 'WASHINGTON DC'),
           ('GPE', 'WASHINGTON, DC'), ('GPE', 'DC'), ('GPE', 'D.C.'),
           ('GPE', 'D.C'))


#%% **********manual merging
def merg_counter(merge_to: Union[Tuple, str], counted: Counter, store=inv_ner):
    if merge_to not in store:
        print(f"{merge_to} not in store")
        return
    c = Counter(dict(store[merge_to]))
    c += counted
    store[merge_to] = list((c).items())


if ('GPE', 'WASHINGTON STATE') in inv_ner:
    state_ids = {x for x, y in inv_ner[('GPE', 'WASHINGTON STATE')]}
else:
    state_ids = set()

merg_counter(
    ('GPE', 'WASHINGTON STATE'),
    Counter(
        {x: y
         for x, y in inv_ner[('GPE', 'WASHINGTON')] if x in state_ids}))
merg_counter(('GPE', 'WASHINGTON, D.C.'),
             Counter({
                 x: y
                 for x, y in inv_ner[('GPE', 'WASHINGTON')]
                 if x not in state_ids
             }))

del inv_ner[('GPE', 'WASHINGTON')]

buffet_ids = {x for x, y in inv_ner[('PERSON', 'WARREN BUFFET')]}
elizabeth_ids = {x for x, y in inv_ner[('PERSON', 'ELIZABETH WARREN')]}

merg_counter(
    ('PERSON', 'WARREN BUFFET'),
    Counter(
        {x: y
         for x, y in inv_ner[('PERSON', 'WARREN')] if x in buffet_ids}))
merg_counter(
    ('PERSON', 'ELIZABETH WARREN'),
    Counter(
        {x: y
         for x, y in inv_ner[('PERSON', 'WARREN')] if x in elizabeth_ids}))

del inv_ner[('PERSON', 'WARREN')]

#%%
import math
RELEVANCE_THRES = math.floor(len(df) * 0.005)

# %% ----inverted df to get which entitites are present with multiple tags
inv_df = pd.DataFrame.from_dict({k: len(v)
                                 for k, v in inv_ner.items()},
                                orient='index',
                                columns=['Len'])
inv_df['Tag'], inv_df['Text'] = zip(*inv_df.index)
inv_df.reset_index()
inv_df.set_index(['Text', 'Tag'], inplace=True)
inv_df.sort_index(inplace=True)
# %% ---- grouping and filtering would be relevants
g = inv_df.reset_index().groupby(['Text']).agg(tc=('Tag', 'count'),
                                               ls=('Len', 'sum'))
g = g[g.tc * g.ls >= RELEVANCE_THRES]
#%% ----- NER that otherwise would not be present
duped = inv_df.loc[inv_df.index.get_level_values(0).isin(
    g.index)][inv_df.Len < RELEVANCE_THRES].copy()

# %% merging counts for better filtering, multi index
duped.reset_index(inplace=True)
duped = duped.merge(g, left_on='Text', right_on='Text')
duped.set_index(['Text', 'Tag'], inplace=True)
duped.sort_index(inplace=True)


# %%
def get_example(text, tag):
    arts = inv_ner[(tag, text)]
    for x in arts:
        yield df.loc[x[0]].Content


# print(get_example('YUKON', 'ORG'))


#%% getting rid of tags
def merge_same_entity(inv: Dict[Tuple, List[Tuple]]) -> Dict[str, List[Tuple]]:
    d = defaultdict(list)
    for tag, text in inv:
        ext = Counter(dict(d[text]))
        ext += Counter(dict(inv[tag, text]))
        d[text] = list((ext).items())
    return dict(d)


merged_inv = merge_same_entity(inv_ner)


#%%
def kwfilterer(k, *must_contain: str, store=merged_inv, stripped=True):
    if k not in store:
        print(f"{k}" not in store)
        return
    targ = 'StrippedContent' if stripped else 'Content'
    res = [(art, c) for (art, c) in store[k]
           if any((m.lower() if stripped else m) in df.loc[art][targ]
                  for m in must_contain)]
    if not res:
        del store[k]
    else:
        store[k] = res


def kwmerger(k, target, store=merged_inv, stripped=True):
    if k not in store:
        print(f"{k}" not in store)
        return
    if target not in store:
        print(f"{target}" not in store)
        return
    targ = 'StrippedContent' if stripped else 'Content'
    res = [(art, c) for (art, c) in store[k]
           if (target.lower() if stripped else target) in df.loc[art][targ]]
    if not res:
        return
    else:
        c = Counter(dict(store[target]))
        c += Counter(dict(res))
        store[target] = list((c).items())


#%% housecleaning after merging
ner_merger('ALEXANDRIA OCASIO - CORTEZ',
           'AOC',
           *[k for k in merged_inv if 'OCASIO' in k],
           store=merged_inv)
kwfilterer('ALEXANDRIA', 'Egypt')

ner_merger('United Nations Climate Change Conference',
           'U.N. CLIMATE CHANGE CONFERENCE',
           'UN CLIMATE CHANGE CONFERENCE',
           'UNITED NATIONS CLIMATE CHANGE CONFERENCE',
           'UN CLIMATE CHANGE CONFERENCE ( COP 25',
           'UN CLIMATE CHANGE CONFERENCE COP 25',
           'CLIMATE CHANGE CONFERENCE COP',
           'UNITED NATIONS CLIMATE CHANGE CONFERENCE 2009',
           "UNITED NATIONS ' CLIMATE CHANGE CONFERENCE",
           *[k for k in merged_inv if 'COP26' in k],
           store=merged_inv)

ner_merger('CORSIA',
           *[
               k for k in merged_inv
               if 'CARBON OFFSETTING AND REDUCTION' in k or 'CORSIA' in k
           ],
           store=merged_inv)

kwfilterer('MCDONALD', "McDonald's", 'McDonald’s', stripped=False)
merged_inv["MCDONALD'S"] = merged_inv.pop("MCDONALD")

kwfilterer('GEORGE', 'bush')
ner_merger('GEORGE BUSH', 'GEORGE', store=merged_inv)

kwfilterer('MARK', 'ZUCKERBERG')
ner_merger('MARK ZUCKERBER', 'MARK', 'ZUCKERBERG', store=merged_inv)

kwfilterer('FRANCIS', 'POPE')
ner_merger('POPE FRANCIS', 'FRANCIS', store=merged_inv)

kwfilterer('GORE', 'AL')
ner_merger('AL GORE', 'GORE', store=merged_inv)

ner_merger('PETE BUTTIGIEG', 'BUTTIGIEG', store=merged_inv)

ner_merger('PARIS AGREEMENT', 'PARIS CLIMATE AGREEMENT', store=merged_inv)
kwmerger('PARIS', 'PARIS AGREEMENT')
kwfilterer('PARIS', 'France')

kwmerger('NEW YORK', 'NEW YORK STATE')
kwfilterer('NEW YORK', 'NEW YORK CITY')
ner_merger('NEW YORK CITY', 'NEW YORK', store=merged_inv)

ner_merger('CENTER FOR DISEASE CONTROL',
           *[
               'CENTERS FOR DISEASE CONTROL',
               'CENTERS FOR DISEASE CONTROL AND PREVENTION',
               'CENTRES FOR DISEASE CONTROL',
               'U.S. CENTERS FOR DISEASE CONTROL AND PREVENTION',
               'U.S. CENTERS FOR DISEASE CONTROL',
               'CENTERS FOR DISEASE CONTROL AND PREVENTION OFFICE',
               'CENTRE FOR DISEASE CONTROL', 'CENTER FOR DISEASE CONTROL',
               'US CENTERS FOR DISEASE CONTROL',
               'US CENTERS FOR DISEASE CONTROL AND PREVENTION',
               'NATIONAL CENTER FOR DISEASE CONTROL',
               'CENTER FOR DISEASE CONTROL AND PREVENTION',
               'CENTER FOR DISEASE CONTROL ’S NATIONAL INSTITUTE',
               'CENTRES FOR DISEASE CONTROL AND PREVENTION',
               'CENTRE FOR DISEASE CONTROL AND PREVENTION',
               'CENTER FOR DISEASE CONTROL(CDC'
           ],
           store=merged_inv)

ner_merger(
    'WORLD HEALTH ORGANIZATION',
    'WHO',
    *[
        k for k in merged_inv
        if 'WORLD HEALTH ORGANISATION' in k or 'WORLD HEALTH ORGANIZATION' in k
    ],
    store=merged_inv)

ner_merger('EMMANUEL MACRON', 'MACRON', store=merged_inv)

ner_merger('ENVIRONMENTAL PROTECTION AGENCY', 'EPA', store=merged_inv)
ner_merger('NATIONAL OCEANIC AND ATMOSPHERIC ADMINISTRATION',
           'NOAA',
           store=merged_inv)
ner_merger('INTERGOVERNMENTAL PANEL ON CLIMATE CHANGE',
           'IPCC',
           store=merged_inv)

ner_merger('GRETA THUNBERG', 'GRETA', 'THUNBERG', store=merged_inv)
ner_merger('DONALD TRUMP',
           'DONALD',
           'TRUMP',
           'PRESIDENT DONALD TRUMP',
           'PRESIDENT TRUMP',
           store=merged_inv)
ner_merger('OBAMA',
           'BARACK OBAMA',
           'BARACK',
           'PRESIDENT OBAMA',
           store=merged_inv)

ner_merger('COVID-19',
           'COVID',
           'SARS-COV2',
           'CORONAVIRUS',
           'COVID19',
           store=merged_inv)  #

ner_merger('UNITED NATIONS', 'U.N.', 'UN', store=merged_inv)  #

for bad_entity in [
        'FRANKLIN', 'ACT', 'GETTY IMAGES IMAGE', 'TELEGRAM',
        'CARBON OFFSETTING AND REDUCTION', 'CHRIS', 'MURPHY', 'THOMAS',
        'MARTIN', 'LEE', 'DOI', 'GUARDIAN AUSTRALIA', 'ENVIRONMENT', 'NATION',
        'SUSTAINABILITY', 'CREATIVE COMMONS ATTRIBUTION - SHARE', 'PH.D.',
        'SCIENTIFIC REPORTS', 'NEW YORKER', 'NPR',
        'THOMSON REUTERS FOUNDATION', 'MICHAEL BLOOMBERG',
        'NATIONAL GEOGRAPHIC', 'LOS ANGELES TIMES', 'INDEPENDENT PREMIUM',
        'GOOGLE NEWS', 'FINANCIAL TIMES', 'BBC', 'TWITTER', 'FACEBOOK',
        'INSTAGRAM', 'UNIVERSITY', 'SOUTH', 'NORTH'
]:
    if bad_entity in merged_inv: del merged_inv[bad_entity]

#%% -------- keeping rare turtles only
PART = math.floor(len(df) * 0.01)  # TODO ezzel jatszani

filtered = {k: v for k, v in merged_inv.items() if len(v) >= PART}
print(len(filtered))

#%%
from nltk.metrics import edit_distance


def lev_sim(s1: str, s2: str):
    return 1 - edit_distance(
        s1, s2, substitution_cost=1.5, transpositions=True) / max(
            len(s1), len(s2))


def leven_mix(l, conf=0.85):
    combos = []
    while True:
        if len(l) == 0:
            return combos
        curr = l[0]
        pairs = [curr]
        passing = []
        for i in range(1, len(l)):
            if lev_sim(curr, l[i]) >= conf:
                pairs.append(l[i])
            else:
                passing.append(l[i])
        if len(pairs) > 1:
            combos.append(pairs)
        l = passing
    return combos


#%%
keys = list(filtered)
to_merge = leven_mix(keys)

for i in to_merge:
    i.sort(key=lambda x: len(x))
    ner_merger(i[0], *i[1:], store=filtered)

with open('../../data/filtered_dict.pt', 'wb') as f:
    pickle.dump(filtered, f, pickle.HIGHEST_PROTOCOL)
