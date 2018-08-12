import os
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

from img_to_vec import Img2Vec
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans


def average(l): return sum(l)/len(l)
 

def compute_vectors(all_files):
    '''
    takes list of paths to all images
    creates {file_name:[float]} 
    list is all floats in feature vector
    '''
    feature_vectors = {}
    for file in all_files:
        try:
            base_image = Image.open(file)
            base_vector = img2vec.get_vec(base_image)
            vector_list = [float(str(x)) for x in np.nditer(base_vector)]
            feature_vectors[file.split('/')[-1]] = vector_list
        except:
            print('ERROR: {}'.format(file))
        else:
            print(file)
    return feature_vectors


def compute_cos_sim(all_files):
    '''
    O(N^2) lol
    '''
    feature_vectors = defaultdict(list)
    for file in all_files:
        print(file)
        base_image = Image.open(file)
        base_vector = img2vec.get_vec(base_image)
        for other_file in all_files:
            if file == other_file:
                continue
            other_image = Image.open(other_file)
            other_vector = img2vec.get_vec(other_image)
               
            cos_sim = cosine_similarity(
                base_vector.reshape((1,-1)),
                other_vector.reshape((1,-1))
                )[0][0]
                 
            feature_vectors[file.split('/')[-1]].append(cos_sim)
    return feature_vectors


def output_results(vectors, file):
    '''
    takes dict of vectors and writes contents to csv file
    '''
    with open(file, 'w') as f:
        for file_name, cos_sim in vectors.items():
            f.write('{},{}\n'.format('row number',str(list(range(len(cos_sim)-1)))[1:-1].replace(' ','')))
            break
        for file_name, cos_sim in vectors.items():
            f.write('{},{}\n'.format(''.join(file_name.split('.')[:-1]),str(cos_sim)[1:-1].replace(' ','')))


def do_PCA(existing_df):
    pca = PCA(n_components=2)
    pca.fit(existing_df)
    existing_2d = pca.transform(existing_df)
    existing_df_2d = pd.DataFrame(existing_2d)
    existing_df_2d.index = existing_df.index
    existing_df_2d.columns = ['PC1','PC2']
    
    ax = existing_df_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(8,4))

    for i, country in enumerate(existing_df.index):
        ax.annotate(country, (existing_df_2d.iloc[i].PC2, existing_df_2d.iloc[i].PC1))

    plt.show()
    return existing_df, existing_df_2d
    
    
def do_KMeans(existing_df, existing_df_2d):
    kmeans = KMeans(n_clusters=10)
    clusters = kmeans.fit(existing_df)
    existing_df_2d['cluster'] = pd.Series(clusters.labels_, index=existing_df_2d.index)

    existing_df_2d.to_csv('us_pc.csv')
    
    axk =existing_df_2d.plot(
        kind='scatter',
        x='PC2',y='PC1',
        c=existing_df_2d.cluster.astype(np.float),
        figsize=(16,8)
    )

    for i, country in enumerate(existing_df.index):
        axk.annotate(country, (existing_df_2d.iloc[i].PC2 + 2, existing_df_2d.iloc[i].PC1 + 2))
    
    plt.show()
    

if __name__ == '__main__':
    PATH = './test_scrape/puppies/'
    img2vec = Img2Vec(cuda=False)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    file = 'puppy_ON.csv'
    if os.path.isfile(Path(file)):
        os.remove(file)
 
    all_files = [PATH + file for file in os.listdir(PATH)]
    feature_vectors = compute_vectors(all_files)
    output_results(feature_vectors, file)
    
    existing_df = pd.read_csv(file, index_col = 0)
    existing_df.index.names = ['file_names']
    existing_df.columns.names = ['r_num']
    
    existing_df, existing_df_2d = do_PCA(existing_df)
    do_KMeans(existing_df, existing_df_2d)    


