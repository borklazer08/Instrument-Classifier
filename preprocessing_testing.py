import numpy as np
import os
import math
import json
import librosa
from sklearn.preprocessing import MultiLabelBinarizer
map=  {
        "cel\t\n":0 , "cla\t\n":1, "flu\t\n" :2 , "gac\t\n":3 , "gel\t\n" :4 , "org\t\n":5 , "pia\t\n" : 6,
        "sax\t\n":7 , "tru\t\n":8 , "vio\t\n":9, "voi\t\n":10
}
data = {
        "labels": [],
        "mfcc": []
    }



DATASET_PATH =("D:\IRMAS training data\IRMAS-TestingData-Part3\Part3",
               "D:\IRMAS training data\IRMAS-TestingData-Part2\IRTestingData-Part2",
               "D:\IRMAS training data\IRMAS-TestingData-Part1\Part1")



JSON_PATH = "D:\IRMAS training data\preprocessing_multi_label.json"
SAMPLE_RATE = 44100

os.chdir(DATASET_PATH[0])
dir_list = os.listdir()


def save_mfcc_multilabel(json_path, num_mfcc=15, n_fft=4096, hop_length=2048, segment_length=1.5):

    for i in range(2):
        os.chdir(DATASET_PATH[i])
        dir_list = os.listdir()

        #save MFCCs
        for i in range(len(dir_list) // 2):

            # iterate through dir
            # open file for labels and file for processing
            #


            num_segments= int(duration//segment_length)
            SAMPLES_PER_TRACK = SAMPLE_RATE * duration

            samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
            num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)


            # process all segments of audio file
            for d in range(num_segments):

                # calculate start and finish sample for current segment
                start = samples_per_segment * d
                finish = start + samples_per_segment

                # extract mfcc
                mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                            hop_length=hop_length)
                mfcc = mfcc.T

                # store MFCCs
                if len(mfcc) == num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())  # to save as a json file
                    data["labels"].append(get_map(i))  #append labels
                    print("{}, segment:{}".format(dir_list[2*i+1], d + 1))



        mlb = MultiLabelBinarizer()
        mlb.fit(data["labels"])
        data["labels"]=mlb.fit_transform(data["labels"])
        data["labels"]=data["labels"].tolist()

          # save MFCCs to json file
        with open(json_path, "w") as fp:
              json.dump(data, fp, indent=4)



#odd are text and even are wav files
def get_map(j):
    # save labels
        f = open(dir_list[2 * j], "r")
        z = f.readlines()
        q = []
        for i in range(len(z)):
            q.append(map[z[i]])

        return q

    # semantic_label = dir_list[0].split("/")[-1]
    # print(semantic_label)


if __name__ =='__main__':
    save_mfcc_multilabel(JSON_PATH)
    type(data["labels"])

