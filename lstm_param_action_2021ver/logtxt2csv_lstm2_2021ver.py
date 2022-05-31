import csv
import os
import sys
import glob

#----#

file_counter = 0

csv_file = open("./log_all_final_lstm2_action_val_2021ver.csv", "w")
csvwriter = csv.writer(csv_file)
#csvwriter.writerow(["video_name","lstmaction_average"])
video_list = []
lstm_list = []
keys = lambda x: (x.split("/")[3].split("_")[1])
file_list = sorted(glob.glob("./text_file/2LSTM_2021ver/BG_*_val_final_key.csv"), key = keys)

for file_name in file_list:
    video_name = file_name.split("/")[3].split("_")[0] + "_" + file_name.split("/")[3].split("_")[1]
    video_list.append(video_name)
    file_counter += 1
    print(video_name)
    with open(file_name) as f:
        reader = csv.reader(f)
        header = next(reader) # header skip
        for line in reader:
            if line[0] == "final":
                weight_average = line[1]
                lstm_list.append(weight_average)
                #csvwriter.writerow([video_name,weight_average])

csvwriter.writerow(video_list)
csvwriter.writerow(lstm_list)
csv_file.close()
print("Convert File_num = {}".format(file_counter))
print("Finish Convert TextFile to csvFile !!")
