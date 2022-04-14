# i dont want to destory my masterpiece `index_conter.py`
# gets the data we have and concvert to a csv of what we need
# [ip_src, ip_dst, weight(cnt_pkt)]
import numpy as np
from tqdm import tqdm

file = '/Volumes/T7 Touch/ITS472/project 2/opt/Malware-Project/BigDataset/IoTScenarios/CTU-IoT-Malware-Capture-8-1/bro/conn.log.labeled'
delim = '\x09' # the separator of each item
start = 6 # INDEX so start line - 1
good_stuff = []
out_csv = open(file + '_formatted.csv', 'w') # get the file name, and add _formatted then give ext '.csv'. file.split('.') # romves after ext
uniques = [] # list for all ip indexes
rows = [2, 4] # INDEX of the column you want from each row
# ^ ip_src ip_dst /*cnt_pkt*/ (not counting the first row with #fields)
data = []
unique_conns = [] # find cleaner way
label = []
train_mask = []

with open(file) as f:
    # skip first few lines
    for i in range(start):
        next(f)
    # get the fields (the first column of csv)
    columns = f.readline().split(delim)[1:]
    # ^ could add '1' to numpy list, but this is smarter
    fields = [ c for c in columns if columns.index(c) in rows ] # for each column in columns, add to fields, if column index is in the needed rows
    fields.append(columns[20].split()[1]) # this is the label
    print('fields:', fields)
    out_csv.write(','.join(fields) + ',cnt' + '\n') # convert to comma separated string and append. should i keep as 2D list, or keep as 1D<string>

    next(f) # skip line with field data types
    for line in (pbar := tqdm(f.readlines())):
        pbar.set_description('Reading File')
        if line.startswith('#close'): # stop before file ends
            break
        line = line.split(delim)
        # have to get uniques
        for i in rows: # i believe all must be abritrary. fastest way?
            if line[i] not in uniques:
                uniques.append(line[i])
        
        # then create list of indexed IP connections
        connection = [ str(uniques.index(c)) for c in line if line.index(c) in rows ]
        data.append(connection) # find way to do in one line :P
        label.append(line[20].split()[1])

    # create a new list of connectinos and the amount of times they connect
    unique_cnt = [] # just create a second array, ratehr than a 2D array of [[conn], cnt], then checking for conn, then cnt++, lss cant check if in only 1 column
    train_mask = [] # list of whether con is malicious, and the %, could be all for naught
    for i, conn in enumerate(pbar := tqdm(data)): # cause using fori is "non-pythonic"
        pbar.set_description('Creating Index')
        if conn not in unique_conns:
            unique_conns.append(conn)
            unique_cnt.append(0)
            train_mask.append(0)
        unique_cnt[unique_conns.index(conn)] += 1 # inc the count by 1
        # ^ super smart. where the conn is stored in the first column, so find its index..
        if label[i] == 'Malicious':
            train_mask[unique_conns.index(conn)] += 1
    for i, mask in enumerate(train_mask):
        train_mask[i] = mask / unique_cnt[i] # converts to weight, right
        
    for i in (pbar := tqdm(range(len(unique_conns)))):
        pbar.set_description('Writing to csv')
        newline = ','.join(unique_conns[i]) + ',' + str(train_mask[i]) + ',' + str(unique_cnt[i]) + '\n'
        # ^ i think this is super cleaver. i think it worked, WHERE IS MY DIPLOMA
        out_csv.write(newline)

print("job done :)")

# # not on fun
# if conn not in unique_conns[0]:
#     unique_conns.append(conn, 0)
# unique_conns[1][unique_conns[0].index(conn)] += 1