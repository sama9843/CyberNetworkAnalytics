import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()#It applies the default seaborn plot theme
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
#import shap
import argparse
import sys

def elbow(df):
    temp = df.copy()
    temp = temp.drop(['exaddr','srcaddr','dstaddr','nexthop'], axis=1)
    X = temp.values

    inertia = []
    k_range = range(1, 10)
    for i in k_range:
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        print("K: ",i," Inertia: ", kmeans.inertia_)
    
    sns.lineplot(x = k_range, y = inertia)
    plt.show()

def kMeans(df):
    dataset = df.copy()
    dataset = dataset.drop(['exaddr','srcaddr','dstaddr','nexthop'], axis=1)
    mat = dataset.values
    # Using sklearn
    km = KMeans(n_clusters=5)
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    results = pd.DataFrame([dataset.index,labels]).T
    return results

def tests(df):
    print(df.head)
    for i in range(20):
        print("Column",i)
        print(df.iloc[:,i].unique())
    #print(df.iloc[:,19].unique())
    #print("Types")
    #print(df.dtypes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--RawDatafile', type=str, required=True, help="The csv raw data file")
    FLAGS = parser.parse_args()
    try:
        raw_data = pd.read_csv(FLAGS.RawDatafile) #Raw data #depends on Netflow file
        print("File Loaded in.")
    except FileNotFoundError:
        print(f"File {FLAGS.RawDatafile} not found.  Aborting")
        sys.exit(1)
    
    tests(raw_data)
    #print("KMeans")
    #tests(kMeans(raw_data))
    #print("Elbow")
    #elbow(raw_data)

main()