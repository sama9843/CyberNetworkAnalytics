import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import argparse
import sys
import plotext
from scipy.spatial.distance import euclidean
from scipy.stats import median_abs_deviation

#some code used from this repo for calculating modified zscore
#https://github.com/isaacarroyov/spotify_anomalies_kmeans-lof

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
    dataset['cluster'] = labels
    return (dataset, km.cluster_centers_)

#gets distance to centroids
def get_dist(df,centroids):
    v = df[:-1]
    label = df[-1]
    centroid = centroids[int(label)]
    dist = euclidean(v,centroid)
    return dist

#gets modified zscore
def get_score(df,medians,mads):
    distance,label = df
    median = medians[int(label)]
    mad = mads[int(label)]
    score = (distance-median)/mad
    return score

#function used for different tests of code
def tests(df,c):
    #prints the inputed dataframes head
    print(df.head)
    dist = np.apply_along_axis(func1d=get_dist , axis=1, arr=df.values, centroids=c)
    print("yep",dist)
    df['distance'] = dist
    print(df.head)
    dist_labels = df[['distance','cluster']].values
    median_0 = np.median(df.query("cluster == 0")['distance'].values)
    median_1 = np.median(df.query("cluster == 1")['distance'].values)
    median_2 = np.median(df.query("cluster == 2")['distance'].values)
    median_3 = np.median(df.query("cluster == 3")['distance'].values)
    median_4 = np.median(df.query("cluster == 4")['distance'].values)
    list_1 = [median_0, median_1, median_2, median_3, median_4]
    mads_0 = median_abs_deviation(df.query("cluster == 0")['distance'].values)
    mads_1 = median_abs_deviation(df.query("cluster == 1")['distance'].values)
    mads_2 = median_abs_deviation(df.query("cluster == 2")['distance'].values)
    mads_3 = median_abs_deviation(df.query("cluster == 3")['distance'].values)
    mads_4 = median_abs_deviation(df.query("cluster == 4")['distance'].values)
    ist_2 = [mads_0, mads_1, mads_2, mads_3, mads_4]
    z_score = np.apply_along_axis(func1d=get_score, arr=dist_labels, axis=1, medians=list_1, mads=list_2)
    df['score'] = z_score
    print(df.head)
    print("Total number of IPs:",len(df.loc[df['score'].abs()>=0]))
    print("IPs with a score greater than 3:",len(df.loc[df['score'].abs()>3]))
    print("IPs with a score greater than 6:",len(df.loc[df['score'].abs()>6]))
    print("IPs with a score greater than 9:",len(df.loc[df['score'].abs()>9]))
    rint("IPs with a score greater than 12:",len(df.loc[df['score'].abs()>12]))
    print("IPs with a score greater than 15:",len(df.loc[df['score'].abs()>15]))
    print("IPs with a score greater than 18:",len(df.loc[df['score'].abs()>18]))
    print(df.loc[df['score']>15])

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
    df, c = kMeans(raw_data)
    tests(df,c)

main()