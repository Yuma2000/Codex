import tkinter as tk
import sys
import numpy as np
import subprocess

import query_embedder

import DebugFunction as df

value = 0

def t2i_retrieval():
    
    value = EditBox.get()
    query = value
    # query = "A person playing tennis"
    print("Query: {}".format(query))
    
    query_emb = query_embedder.get_query_emb(
        query,
        "data/models/cc+mscoco_precomp/model_best.pth.tar",
        "data/vocab/cc+mscoco_precomp_vocab_src.txt"
    )
    
    shotInfos = {}
    shotInfos2 = []  # Used for result visualisation
    nbShots = 0
    with open("data/TRECVID/shotInfo.txt", "r") as fin:
        for line in fin:
            src = line[:-1].split(' ')
            if src[0] not in shotInfos:
                shotInfos[src[0]] = 0
            shotInfos[src[0]] += 1
            shotInfos2.append( (src[0], src[1]) )
            nbShots += 1
    print(">> # of shots = {}".format(nbShots))
    nbVideos = len(shotInfos)
    print(">> # of videos = {}".format(nbVideos))
    
    img_embs_root_dir = "data/TRECVID/shot_embs/"
    sims_all = np.empty(nbShots)
    offset = 0
    for vid in range(nbVideos):
        vid_str = str(vid+1).zfill(5)
        img_embs_filename = img_embs_root_dir + vid_str + ".npy"
        if vid % 1000 == 0:
            print(
                "### Load precomputed image embeddings from {}"
                .format(img_embs_filename)
            )
        img_embs = np.load(img_embs_filename)
        sims = np.dot(query_emb, img_embs.T)
        assert sims.shape[1] == shotInfos[vid_str],\
            "ERROR: Inappropriate # of shots ({} vs. {})".format(sims.shape[1], shotInfos[vid_str])
        sims_all[offset:offset+shotInfos[vid_str]] = sims[0, :]
        offset += shotInfos[vid_str]
    
    rank = np.argsort(sims_all)
    rank = rank[::-1]
    # df.set_trace()
    
    keyframeRootDir = "data/TRECVID/keyframes/"
    videoRootDir = "data/TRECVID/videos/"
    outputFilename = "result.html"
    nbOutputs = 20
    nbOutputs_one_row = 5    
    print(">> Output the result into {}".format(outputFilename))
    with open(outputFilename, "w") as fout:
        
        print("<html><head><title>{}</title></head>\n<body>\n".format(outputFilename), file=fout)
        for i in range(nbOutputs):
            
            if i % nbOutputs_one_row == 0:
                print("<table cellpadding=5><tr>\n", file=fout)
            
            keyframeFilename = keyframeRootDir + shotInfos2[rank[i]][0] + "/shot"\
                + shotInfos2[rank[i]][0] + "_" + str( int(shotInfos2[rank[i]][1]) + 1 ) + "_RKF.png"
            videoFilename = videoRootDir + shotInfos2[rank[i]][0] + "/shot"\
                + shotInfos2[rank[i]][0] + "_" + str( int(shotInfos2[rank[i]][1]) + 1 ) + ".webm"
            
            if i % nbOutputs_one_row == 0:
                print(
                    "--- {}th {}-{} ({})"
                    .format(i, shotInfos2[rank[i]][0], shotInfos2[rank[i]][1], sims_all[rank[i]])
                )
                print(keyframeFilename)
            
            print(
                "<td><table align=middle><tr align=middle><td><a href=\"{}\"><img src=\"{}\" width=250></a></td></tr><tr align=middle><td>"
                .format(keyframeFilename, keyframeFilename), file=fout
            )
            print(
                "<font size=1>{}:{}-{}<br>sim:{:.5f}</font></td></tr></table></td>\n"
                .format(i+1, shotInfos2[rank[i]][0], shotInfos2[rank[i]][1], sims_all[rank[i]]),
                file=fout
            )
            
            if i % nbOutputs_one_row == (nbOutputs_one_row - 1):
                print("</tr></table>\n \n", file=fout)
        
        if nbOutputs % nbOutputs_one_row != 0:
            print("</tr></table>\n \n", file=fout)
        print("</body></html>", file=fout)
    
    subprocess.call("firefox result.html", shell=True)
            

root = tk.Tk()

root.title("demo")
root.geometry("800x230")

label1 = tk.Label(text='Enter a query! (English only)',font=('ＭＳ ゴシック', 20))
label2 = tk.Label(text='ex.1) a person sleeping',font=('ＭＳ ゴシック', 20))
label3 = tk.Label(text='ex.2) a race car driver racing a car',font=('ＭＳ ゴシック', 20))
label4 = tk.Label(text='ex.3) a shirtless man standing up or walking outdoors',font=('ＭＳ ゴシック', 20))
label5 = tk.Label(text='',font=('ＭＳ ゴシック', 20))

label1.pack(anchor=tk.W) 
label2.pack(anchor=tk.W)
label3.pack(anchor=tk.W)
label4.pack(anchor=tk.W)
label5.pack(anchor=tk.W)
#label.place(x = 150, y = 228)

EditBox = tk.Entry(width = 60,font=('',20))
EditBox.pack()
value = EditBox.get()

Button = tk.Button(text='search', width = 60,font=('ＭＳ ゴシック',20),command=lambda:t2i_retrieval())
Button.pack()
root.mainloop()

# if __name__ == '__main__':
#    t2i_retrieval()

