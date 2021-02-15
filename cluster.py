import os
import re
import glob
import string
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import email
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')


class Cluster():
    def __init__(self,data: pd.DataFrame, num_clusters: int):
        self.data = data
        self.stopwords_list = stopwords.words('english')

        # create KMeans model with num_clusters input
        self.model = KMeans(n_clusters=num_clusters,max_iter=50,init='k-means++',n_init=1,random_state=0)

        self.vectorizer = TfidfVectorizer(analyzer='word',stop_words=self.stopwords_list, ngram_range=(1,3))
        # vectorize the body of emails in provided data 
        self.vector = self.vectorizer.fit_transform(data['body'])
        self.labels = self.model.fit_predict(self.vector)

        self.array = self.vector.toarray()
        self.dense_vec = self.vector.todense()
        # Use PCA to improve data visualization
        self.pca = PCA(n_components=2).fit(self.dense_vec)
        self.transform = self.pca.transform(self.dense_vec)

    def plot_data(self,color: np.ndarray):
        label = ["#ec0000","#007100","#006f71","#ffa933","#f333ff",'#fff933','#43ef08','#2dfcff']
        # color = [label[i] for i in self.labels]
        scatter = plt.scatter(self.transform[:, 0], self.transform[:, 1], c=color)
        ls = np.unique(color)
        handles = [plt.Line2D([],[],marker="o", ls="", 
                              color=scatter.cmap(scatter.norm(yi))) for yi in ls]
        plt.legend(handles, ls)

    # alternate plotting function
    def dbscan(self,eps: float) -> np.ndarray:
        dbscan = DBSCAN(
         eps = eps, 
         metric='euclidean', 
         n_jobs = -1)
        clusters = dbscan.fit_predict(self.transform)
        self.plot_data(clusters)
        return clusters

    # Function to compute top features for given cluster
    def get_top_features(self,n_feats: int) -> pd.DataFrame:
        prediction = self.model.predict(self.vector)
        labels = np.unique(prediction)
        dfs = []
        for label in labels:
            id_temp = np.where(prediction==label)
            x_means = np.mean(self.array[id_temp],axis=0)
            sorted_means = np.argsort(x_means)[::-1][:n_feats]
            features = self.vectorizer.get_feature_names()
            best_features = [(features[i], x_means[i]) for i in sorted_means]
            df = pd.DataFrame(best_features,columns=['features','score'])
            dfs.append(df)
        return dfs

    def get_common_words(self) -> nltk.probability.FreqDist:
        # Collect most frequently used words in the distribution
        w = []
        for email in self.data.body:
            w.append(self.data.body)

        misc_punctuation = ['...',"``","''",'e','l','w','k','n','v','b','f','r','~~','//']
        stop = self.stopwords_list + list(string.punctuation) + misc_punctuation
        tokenized = word_tokenize(str(w))
        useful = [word for word in tokenized if word not in stop]
        return nltk.FreqDist(useful)

    def find_optimal_k(self):
        # Find the optimal number of clusters using elbow method
        square_distances = []
        for k in range(1,10):
            if k % 5 == 0:
                print('.',end=' ')
            km = KMeans(n_clusters=k,max_iter=100,init='k-means++',n_init=1)
            km = km.fit(self.transform)
            square_distances.append(km.inertia_)
            
        plt.plot(range(1,10),square_distances,'bx-')
        plt.xlabel('k')
        plt.ylabel('square distances')
        plt.title('optimal k')










