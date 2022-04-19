import csv
import numpy as np

in_file = 'conn.log.labeled_formatted2.csv'
out_file = open('conn.log.labeled_reformatted2.csv', 'w')
unlinked = []

with open(in_file) as f: # for now convert to index list by hand
        unlinked=[tuple(line) for line in csv.reader(f)]
fields = list(unlinked[:1][0])
unlinked=unlinked[1:] # skip the labels

unzipped = list(zip(*unlinked))
ips = [list(map(int, list(unzipped[0]))), list(map(int, list(unzipped[1])))] # seems i could do this better

n_nodes = np.max(ips) + 1 # not as fun, but more efficienct

out_file.write(fields[0] + ',' + 'conn' + ',' + fields[1] + ',' + fields[2] + ',' + fields[3] + '\n')
for i, (src, dst, isMal, cnt) in enumerate(unlinked): # sure
    out_file.write(src + ',' + str(n_nodes+i) + ',' + dst + ',' + isMal + ',' + cnt + '\n')

print('job done :P')