import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
#import shap
import argparse
import sys
import plotext

#elbow plot used to determine amounts of clusters for k-means
def elbow(df):
    temp = df.copy()
    #gets rid of non numerical data(ip addresses)
    temp = temp.drop(['exaddr','srcaddr','dstaddr','nexthop'], axis=1)
    X = temp.values
    inertia = []
    #test k-values of 1-19
    k_range = range(1, 20)
    for i in k_range:
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        print("K: ",i," Inertia: ", kmeans.inertia_)
    
    #creates elbow plot and displays in terminal
    plotext.plot(k_range,inertia)
    plotext.show()

#runs k-means on dataframe
def kMeans(df):
    dataset = df.copy()
    #drops columns that have the same value for all entries, also drops dstaddr as not used
    dataset = dataset.drop(['exaddr','engine_type','dstaddr','nexthop','input','tos','src_mask','dst_mask','src_as','dst_as'],axis=1)
    #groups netflows by ip address and takes the mean of all of the netflows for each column
    dataset = dataset.groupby(['srcaddr']).mean()
    mat = dataset.values
    # Using sklearn
    km = KMeans(n_clusters=5)
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    results = pd.DataFrame([dataset.index,labels]).T
    return results

#function used for different tests of code
def tests(df):
    #prints the inputed dataframes head
    print(df.head)
    #used to print out all column names and unique values per column
    #for i in range(df.shape[1]):
    #    print(df.columns.values[i])
    #    print(df.iloc[:,i].unique())
    #print("Types");
    #print(df.dtypes);
    
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
    
    #tests(raw_data)
    print("Elbow")
    elbow(raw_data)
    print("KMeans")
    tests(kMeans(raw_data))
    

main()