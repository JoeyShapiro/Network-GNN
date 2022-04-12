# i dont want to destory my masterpiece `index_conter.py`
# gets the data we have and concvert to a csv of what we need
# [ip_src, ip_dst, weight(cnt_pkt)]
import numpy as np

file = '/Volumes/T7 Touch/ITS472/project 2/opt/Malware-Project/BigDataset/IoTScenarios/CTU-Honeypot-Capture-4-1/bro/conn.log.labeled'
delim = '\x09' # the separator of each item
start = 6 # INDEX so start line - 1
good_stuff = []
out_csv = open(file.split('.')[0] + '_formatted.csv', 'w') # get the file name, and add _formatted then give ext '.csv'
uniques = [] # list for all ip indexes
rows = [2, 4] # INDEX of the column you want from each row
# ^ ip_src ip_dst /*cnt_pkt*/ (not counting the first row with #fields)
data = []
unique_conns = [] # find cleaner way

with open(file) as f:
    # skip first few lines
    for i in range(start):
        next(f)
    # get the fields (the first column of csv)
    columns = f.readline().split(delim)[1:] 
    # ^ could add '1' to numpy list, but this is smarter
    fields = [ c for c in columns if columns.index(c) in rows ] # for each column in columns, add to fields, if column index is in the needed rows
    print('fields:', fields)
    out_csv.write(','.join(fields) + ',cnt' + '\n') # convert to comma separated string and append. should i keep as 2D list, or keep as 1D<string>

    next(f) # skip line with field data types
    for line in f.readlines():
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

    # create a new list of connectinos and the amount of times they connect
    unique_cnt = [] # just create a second array, ratehr than a 2D array of [[conn], cnt], then checking for conn, then cnt++, lss cant check if in only 1 column
    for conn in data:
        if conn not in unique_conns:
            unique_conns.append(conn)
            unique_cnt.append(0)
        unique_cnt[unique_conns.index(conn)] += 1 # inc the count by 1
        # ^ super smart. where the conn is stored in the first column, so find its index..
        
    for i in range(len(unique_conns)):
        newline = ','.join(unique_conns[i]) + ',' + str(unique_cnt[i]) + '\n'
        # ^ i think this is super cleaver. i think it worked, WHERE IS MY DIPLOMA
        out_csv.write(newline)

print("job done :)")

# # not on fun
# if conn not in unique_conns[0]:
#     unique_conns.append(conn, 0)
# unique_conns[1][unique_conns[0].index(conn)] += 1