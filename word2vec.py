import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import argparse
import numpy as np
from sklearn.cluster import KMeans
import plotext

# get sample
def sample(df):
    sampled = df.sample(frac=0.01, random_state=8)
    return sampled

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

df = sample(raw_data)

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

# Convert the dataframe to a graph representation
graph = nx.from_pandas_edgelist(df, source='srcaddr', target='dstaddr')

# Train the node2vec model

# node2vec returns a trained word embedding model, which can be used to obtain embeddings for individual nodes
# in a graph. These embeddings are typically dense, low-dimensional vectors that capture the semantic or structural
# relationships between nodes in the graph.

# In the context of node2vec, generating walks means creating sequences of nodes that simulate a random walk
# in the graph. This is done by starting at a random node and then at each step, selecting the next node based
# on a transition probability function that takes into account the local neighborhood structure of the graph.

# By generating multiple such random walks, we obtain a set of sequences that represent different paths through
# the graph, allowing the node2vec model to learn the embeddings of the nodes based on their co-occurrence patterns
# in these paths.

node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, p=0.5, q=2, workers=8)
model = node2vec.fit(window=10, min_count=1, batch_words=4, callbacks=[ProgressLogger()])

# Get the embeddings for all nodes in the graph
embeddings = {}
for node in graph.nodes():
    embeddings[node] = model.wv[node]

# Perform kmeans on the embeddings
# Convert the embeddings into a numpy array
X = np.array(list(embeddings.values()))

# Define the number of clusters
elbow(df)

# Train the k-means model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get the cluster labels for each node
cluster_labels = {}
for i, node in enumerate(graph):
    cluster_labels[node] = kmeans.labels_[i]

# Print the number of nodes in each cluster
for i in range(num_clusters):
    print(f"Cluster {i}: {sum(cluster_labels == i)} nodes")

#Analyze each cluster
for i in range(num_clusters):
    cluster_nodes = [node for node, label in zip(embeddings.keys(), kmeans.labels_) if label == i]
    print(f"Cluster {i} - number of nodes: {len(cluster_nodes)}")
    top_sources = df[df['srcaddr'].isin(cluster_nodes)]['srcaddr'].value_counts().nlargest(5)
    top_targets = df[df['dstaddr'].isin(cluster_nodes)]['dstaddr'].value_counts().nlargest(5)
    print(f"Top sources:\n{top_sources}")
    print(f"Top targets:\n{top_targets}\n")
