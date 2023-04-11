# Importing modules and packages
import pandas as pd
import networkx as nx 
from node2vec import Node2Vec
from gensim.models.callbacks import CallbackAny2Vec
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from collections import Counter
from sklearn.neighbors import LocalOutlierFactor

# define constants
SAMPLE_SIZE = 0.005
RANDOM_STATE = 8
MAX_K_TESTED = 10

# get sample from data
def sample(df, sample_size):
    sampled = df.sample(frac=SAMPLE_SIZE, random_state=RANDOM_STATE)
    return sampled

# Convert the dataframe to a graph representation
def graph_rep(df):
    graph = nx.from_pandas_edgelist(df, source='srcaddr', target='dstaddr')
    return graph

#elbow plot used to determine amounts of clusters for k-means
def elbow(df):
    temp = df.copy()
    #gets rid of non numerical data(ip addresses)
    temp = temp.drop(['exaddr','srcaddr','dstaddr','nexthop'], axis=1)
    T = temp.values
    inertia = []

    #test k-values of 1-19
    k_range = range(1, MAX_K_TESTED)
    for i in k_range:
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = RANDOM_STATE)
        kmeans.fit(T)
        inertia.append(kmeans.inertia_)
        print("K: ",i," Inertia: ", kmeans.inertia_)

    #creates elbow plot and displays in terminal
    #plotext.plot(k_range,inertia)
    #plotext.show()

    return (inertia.index(min(inertia)) + 1)

#runs k-means on dataframe
def kMeans(training_data, k):
    # Train the k-means model
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    kmeans.fit(training_data)
    return kmeans

def main():
    # Load the dataframe
    parser = argparse.ArgumentParser()
    parser.add_argument('--RawDatafile', type=str, required=True, help="The csv raw data file")
    FLAGS = parser.parse_args()

    try:
        raw_data = pd.read_csv(FLAGS.RawDatafile) #Raw data #depends on Netflow file
        print("File Loaded in.")

    except FileNotFoundError:
        print("File not found. Aborting")
        sys.exit(1)
    
    #raw_data = pd.read_csv('ft-v05.2023-04-11.060000-0600.csv')

    df = sample(raw_data, SAMPLE_SIZE)

    # Define a callback to print progress during training
    # The progress logger is a callback function used during the training of a Node2Vec model that prints out the loss
    # value after each epoch. The loss value is a measure of how well the model is able to predict the context words
    # given a target word. By printing out the loss value, the progress logger can provide feedback on the training
    # progress of the model, and allow for adjustments to be made to the training parameters if necessary.
    class ProgressLogger(CallbackAny2Vec):
        def __init__(self):
            self.epoch = 0
            self.loss = 0
        def on_epoch_end(self, model):
            self.epoch += 1
            print(f"Epoch {self.epoch} - loss: {self.loss / model.corpus_total_words:.4f}")
            self.loss = 0

    print("Graph representation")
    graph = graph_rep(df)



    # node2vec returns a trained word embedding model, which can be used to obtain embeddings for individual nodes
    # in a graph. These embeddings are typically dense, low-dimensional vectors that capture the semantic or structural
    # relationships between nodes in the graph.

    # In the context of node2vec, generating walks means creating sequences of nodes that simulate a random walk
    # in the graph. This is done by starting at a random node and then at each step, selecting the next node based
    # on a transition probability function that takes into account the local neighborhood structure of the graph.

    # By generating multiple such random walks, we obtain a set of sequences that represent different paths through
    # the graph, allowing the node2vec model to learn the embeddings of the nodes based on their co-occurrence patterns
    # in these paths.

    print("Model")
    # Train the node2vec model
    node2vec = Node2Vec(
        graph,
        dimensions=64,
        walk_length=30,
        num_walks=10,
        workers=10,
        p=0.25,
        q=4
    )

    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    # Get the embeddings for all nodes in the graph
    embeddings = {}
    for node in graph:
        embeddings[node] = model.wv[node]

    # Compute the cosine similarity between the embeddings of all pairs of nodes
    similarity_matrix = cosine_similarity(list(embeddings.values()))
    print("\nSimilarity Matrix")
    print(similarity_matrix)

    # Determine the number of clusters
    # Use the elbow method defined
    print("\nElbow")
    k = elbow(df)
    print(f"Number of clusters: {k}\n")


    # Perform spectral clustering on the similarity matrix
    # The idea behind spectral clustering is to first construct a similarity matrix that
    # captures the pairwise similarities between the data points, and then to use the
    # eigenvectors of this matrix to embed the data points into a low-dimensional space.
    # Once the data points are embedded in this low-dimensional space, a clustering algorithm
    # is applied to partition the data points into clusters.
    
    spectral = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='discretize')
    cluster_labels = spectral.fit_predict(similarity_matrix)

    # Print the cluster labels for each node (tells us which IP's belong to which cluster)
    #for node, label in zip(embeddings.keys(), cluster_labels):
    #    print(f"Node {node} belongs to cluster {label}")

    # Prints out how many nodes per cluster
    counts = Counter(cluster_labels)
    for cluster_label, count in counts.items():
        print(f"Cluster {cluster_label}: {count} nodes")

    # Anomaly detection
    # Calculate the LOF scores for each node

    # One common approach for anomaly detection is to use the Local Outlier Factor (LOF) algorithm,
    # which measures the local density of a point relative to its neighbors to identify points that
    # have a significantly lower density.

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    lof_scores = lof.fit_predict(list(embeddings.values()))

    # Print the nodes with the highest LOF scores

    # A LOF score of 1 indicates that the node is very far away from its neighbors and does not
    # fit well within the local neighborhood structure of the graph. This could indicate that
    # these nodes are potentially malicious or exhibit behavior that is significantly different
    # from the rest of the network. It is important to investigate these nodes further to
    # determine whether they are indeed anomalous and require further action.

    print("")
    for node, score in sorted(zip(graph.nodes(), lof_scores), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Node {node} has an LOF score of {score:.4f}")


    # Convert the embeddings into a numpy array
    X = np.array(list(embeddings.values()))

    # Perform kmeans on the embeddings
    print("\nKMeans")
    kmeans = kMeans(X, k)

    # Get the cluster labels for each node
    cluster_labels = kmeans.labels_

    # Print the number of nodes in each cluster
    for i in range(k):
        print(f"Cluster {i+1}: {sum(cluster_labels == i)} nodes")

    #Analyze each cluster
    print("")
    for i in range(k):
        cluster_nodes = [node for node, label in zip(embeddings.keys(), kmeans.labels_) if label == i]
        print(f"Cluster {i} - number of nodes: {len(cluster_nodes)}")
        top_sources = df[df['srcaddr'].isin(cluster_nodes)]['srcaddr'].value_counts().nlargest(5)
        top_targets = df[df['dstaddr'].isin(cluster_nodes)]['dstaddr'].value_counts().nlargest(5)
        print(f"Top sources:\n{top_sources}")
        print(f"Top targets:\n{top_targets}\n")

main()


