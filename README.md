# Graph Neural Network (GNN) for Network Traffic
## GOAL
Find a network that has malicious traffic in it.<br>
### Most of the files here are for testing purposes

## INSTALLING
I am still not sure how to simply say what libraries are needed.<br>
In this repo, there are several files with information on what is needed.<br>
For starters, I generated a `package-list.txt`, this is auto-generated from `conda`, so if you know how to use that, then go for it, and best of luck<br>
I also create a `setup.sh`, by hand, which is a list of commands and things i did to get the env setup. If you dont have a unix-based system, you may need to change some commands around. But it should have more or less everything, and how i did it.<br>

### Note
This is not an actual `sh` file, just a nice place to put commands. I dont know if it will work, but i wouldnt really try it.<br><br>
But simply put, this uses `stellar graph`, `python 3.8`, `tensor flow`, `torch`, and `networkx`.<br>
Good luck figuring out the install, most of the demos I found dont really explain this part.<br>

## RUNNING
Only a few files are really needed, the rest is just for me, or to show others.<br>
### First you need the data
You could use a pcap file, this can be seen in the `OLDREADME.md`, or you can just download one.<br>
But really you need a dataset file, the style I am using now uses a special data format to create the csvs.<br>
The important part is the data uses the proper format.<br>
`ip_src_i, ip_dst_i, isMalware, pkt_cnt` (the 'i' in src and dst is there unique id)<br>
`int, int, float, int`<br>
`0, 1, 0.0, 100`<br>
`0, 2, 0.9, 50`<br>
where 0.0 is 0% of the packets are malicious, and 1.0 is 100% of packets are malicious<br>

#### Note
This format will change over time, and I may forget to update it.<br>
Please refer to `csver.py`, to see my style format
### Then convert them
It should follow the format, and `csver.py` will either convert or help you the data.<br>
The formatted files should go in a folder. The folder is marked in `DGCNN.py`, and will need a special format:<br>
`dataset` (you will need to change the path name in the GNN)<br>
`-mal` (for mal network csvs)<br>
`-bon` (for good network csvs)<br>
### Then Run GNN
The GNN to use is `DGCNN.py`, at least as of now. give it the folder of datsets and watch it go.<br>
It will show a sample graph of the first network, with a heat map.<br>
It will then convert the data.<br>
Then train.<br>
Then test.<br>
Then predict.<br>

## ABOUT
