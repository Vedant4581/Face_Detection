import numpy as np
import pandas as pd
import cv2

import redis

#insightface

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

#connect to redis client
#connection to database
#Connect to redis client

hostname='redis-19868.c267.us-east-1-4.ec2.cloud.redislabs.com'
portnumber=19868
password='ElZFvkWqGrcayZt7SMmnmknmc9rX5nUC'

r=redis.StrictRedis(host=hostname,
                    port=portnumber,
                    password=password)

#configure face analysis

faceapp=FaceAnalysis(name='buffalo_l',
                     root='insightface_model',
                     providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0,det_size=(640,640),det_thresh=0.5)

#ML Search Algorithm

def ml_search_algorithm(dataframe,feature_column,test_vector,name_roll=['Name','Roll'],thresh=0.5):
    #step-1: take the dataframe(collection of data)
    dataframe=dataframe.copy()
    #step-2: Index face embeding from the dataframe
    x_list=dataframe[feature_column].tolist()
    x=np.asarray(x_list)
    #step-3: Calculate cosine similarity
    similar=pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr=np.array(similar).flatten()
    dataframe['cosine']=similar_arr
    #step-4: filter the data
    data_filter=dataframe.query(f'cosine>={thresh}')
    if len(data_filter)>0:
        #step-5: get the person name
        data_filter.reset_index(drop=True,inplace=True)
        argmax=data_filter['cosine'].argmax()
        person_name,person_roll=data_filter.loc[argmax][name_roll]
    else:
        person_name='Unknown'
        person_roll='Unknown'

    return person_name,person_roll


def face_prediction(test_image,dataframe,feature_column,name_roll=['Name','Roll'],thresh=0.5):
    #step-1: take the test image and apply to insight face
    results=faceapp.get(test_image)
    test_copy=test_image.copy()
    #step-2: use for loop and extract each embedding and pass to ml_search_algorithm

    for res in results:
        x1,y1,x2,y2=res['bbox'].astype(int)
        embeddings=res['embedding']
        person_name,person_roll=ml_search_algorithm(dataframe,feature_column,test_vector=embeddings,name_roll=name_roll,thresh=thresh)
        cv2.rectangle(test_copy,(x1,y1),(x2,y2),(0,255,0))
        text_gen=person_name
        if person_name=='Unknown':
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.35,(0,0,255),1)
        else:
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.35,(0,255,0),1)
    return test_copy