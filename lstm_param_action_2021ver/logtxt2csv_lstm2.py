import csv
import os
import sys
import glob

#----#

file_counter = 0

csv_file = open("./log_all_final_lstm2_actionver.csv", "w")
csvwriter = csv.writer(csv_file)
csvwriter.writerow(["video_name","lstmaction_average"])

for file_name in glob.glob("./text_file/2LSTM/BG_*_final_key.txt"):
    video_name = file_name.split("/")[3].split("_")[0] + "_" + file_name.split("_")[2]
    file_counter += 1
    print(video_name)
    with open(file_name) as f:
        next(f)
        for line in f:
            if line.split(",")[0] == "final":
                weight_average = line.split(",")[1]
                csvwriter.writerow([video_name,weight_average])

csv_file.close()
print("Convert File_num = {}".format(file_counter))
print("Finish Convert TextFile to csvFile !!")
