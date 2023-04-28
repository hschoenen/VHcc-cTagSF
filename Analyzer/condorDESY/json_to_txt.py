# copy content of a json filelist into a txt filelist
# python json_to_txt.py file.json file.txt

import os,sys

json_filename=sys.argv[1]
json_file=open(json_filename,'r')
txt_filename=sys.argv[2]
txt_file=open(txt_filename,'w')

first_line=1
for i,line in enumerate(json_file):
    if "root" in line:
        if first_line==0:
            txt_file.write("\n")
        file=line.split('"')[1]
        txt_file.write(file)
        first_line=0