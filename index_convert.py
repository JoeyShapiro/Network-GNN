# first run
# /Applications/Wireshark.app/Contents/MacOS/tshark -r test.pcap -T fields -e ip.src -e ip.dst -E separator=, > test.csv
# to get a csv

file = 'bigFlows.csv'

file_in = open(file)
file_out = open('formatted_' + file, 'w')
uniques = []

# clean file
lines = file_in.readlines()
for line in lines:
    line = line.split(',')#[:2] <-  redundant
    if (len(line) < 2): # skip if not right (sometimes line is just ",")
        continue
    # check if either in tuple unique IP for later
    if line[0] not in uniques:
        uniques.append(line[0]) # do all in same loop
    if line[1].strip() not in uniques:
        uniques.append(line[1].strip()) # i think i strip
    newline = str(uniques.index(line[0])) + ',' + str(uniques.index(line[1].strip())) + '\n' # only keep first 2 (i think tshark bugs out). newline to keep clean
    # ^ i think this is super cleaver. i think it worked, WHERE IS MY DIPLOMA
    file_out.write(newline) # put the clean thing in the new file

file_in.close() # close them for safety or whatever
file_out.close()