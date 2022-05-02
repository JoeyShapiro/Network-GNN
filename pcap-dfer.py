import pandas as pd
from tqdm import tqdm

df = pd.DataFrame()

file = '/Users/oniichan/Downloads/SUEE1.csv'

start = 0
delim = ','
rows = [0, 1, 2]
data = {}
unique_IPs = []

csv = pd.read_csv(file)

with open(file) as f:
    for i in range(start):
        next(f)
    
    columns = f.readline().split(delim)
    fields = [ c for c in columns if columns.index(c) in rows ]
    print('fields', fields)
    next(f) # skip something

    for field in fields:
        data[field] = [] # i think array

    for line in (pbar := tqdm(f.readlines())):
        pbar.set_description('Reading File')
        if line.startswith('#close'): # stop before file ends
            break
        line = line.split(delim)
        row = []
        for i in rows:
            row.append(line[i].strip())

        if row[0] not in unique_IPs:
            unique_IPs.append(row[0])
        if row[1] not in unique_IPs:
            unique_IPs.append(row[1])

        row[0] = unique_IPs.index(row[0]) # good for now, look up how to do with df
        row[1] = unique_IPs.index(row[1])

        # cols[0].append('row[0]')# appeneds first row to this
        for i, field in enumerate(fields): # this works and is cleaner
            data[field].append(row[i])

df = pd.DataFrame(data=data)
print(df)

for ip in unique_IPs:
    df.replace(ip, unique_IPs.index(ip)) # sure, anywhere i guess

# print(df.loc[(df['id.resp_p'] == '123') & (df['id.resp_h'] == 1) & (df['id.orig_h'] == 0)])

df = df.reset_index() # ?
unique_cons = []
unique_cons_cnts = [] # find how to make 1 thing

for index, row in (pbar := tqdm(df.iterrows())):
    pbar.set_description('Indexing rows')
    unindexed = row.tolist()[1:]
    if unindexed not in unique_cons:
        unique_cons.append(unindexed)
        unique_cons_cnts.append(0)
    unique_cons_cnts[unique_cons.index(unindexed)] += 1

print("creating frequency map")
ndf = pd.DataFrame(index=fields)
for unique in (pbar := tqdm(unique_cons)):
    pbar.set_description('Mapping frequency')
    what = pd.DataFrame(pd.Series(unique, index=fields).to_dict(), index=[0])
    ndf = pd.concat([ndf, what], ignore_index=True) # YOU HAVE TO SAVE IT
ndf = ndf.iloc[4:, : ] # because first 4 are nan
ndf['cnt'] = unique_cons_cnts
# convert from float back to int
ndf[fields[0]] = ndf[fields[0]].astype(int) # find better way, seems to built for
ndf[fields[1]] = ndf[fields[1]].astype(int)

print(ndf)
ndf.to_csv(file+'.csv', index=False) # better with no index, i guess
print('Converted to csv')