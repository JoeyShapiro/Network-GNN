# Graph Neural Network (GGN) for Network Traffic
## Finds anomalies based on network traffic in a pcap file
### Most of the files are for testing purposes
## INSTALLING
Im not sure how to cleany and simply say what packages are installed<br>
This uses `python 3.9, dgl, pandas, and networkx` as of right now.<br>
Because as of right now im not sure, i added a `packages-list.txt`<br>
This was auto generated by conda, so it should have everything.<br>
Good luck. People never seem to say their packages or versions they run.<br>
## RUNNING
Only a few files are really needed.<br>
### First you need a pcap file.
This can be found many ways.<br>
You can find one from one of the links in the useful_link.txt.<br>
Or you can simply use your own<br>
### Then convert the pcap to a csv
This can be done with the command at the top of `index_convert.py`.<br>
It is a `tshark` command, this comes with `Wireshark`.<br>
You can also use your own method.<br>
You can choose any format you want, but the code is currently meant for the format:<br>
`src,dst`
`192.168.10.10,8.8.8.8`
This will give you a csv file.<br>
### Then run index_convert.py
Change `file` to the csv file you want to convert<br>
This will return a `formatted_file.csv`<br>
Use this for the next step<br>
### Then use network.py
First change the csv to open on line 7 with your csv file.<br>
Then run the script.<br>
This will give you output and display a graph of the connections.<br>