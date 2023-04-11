import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import argparse
import numpy as np
from sklearn.cluster import KMeans

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
class ProgressLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss = 0
    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"Epoch {self.epoch} - loss: {self.loss / model.corpus_total_words:.4f}")
        self.loss = 0

# Convert the dataframe to a graph representation
graph = {}
for _, row in df.iterrows():
    source = row['srcaddr']
    target = row['dstaddr']
    if source not in graph:
        graph[source] = []
    if target not in graph:
        graph[target] = []
    graph[source].append(target)
    graph[target].append(source)

# Train the node2vec model
model = Word2Vec(
    sentences=graph.values(),
    vector_size=64,
    window=10,
    min_count=1,
    workers=8,
    sg=1,
    hs=0,
    negative=5,
    ns_exponent=0.75,
    epochs=50,
    compute_loss=True,
    callbacks=[ProgressLogger()]
)

# Get the embeddings for all nodes in the graph
embeddings = {}
for node in graph:
    embeddings[node] = model.wv[node]

# Convert the embeddings into a numpy array
X = np.array(list(embeddings.values()))

# Define the number of clusters
num_clusters = 10

# Train the k-means model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Get the cluster labels for each node
cluster_labels = kmeans.labels_

# Print the number of nodes in each cluster
for i in range(num_clusters):
    print(f"Cluster {i}: {sum(cluster_labels == i)} nodes")

for i in range(5):
    cluster_nodes = [node for node, label in zip(embeddings.keys(), kmeans.labels_) if label == i]
    print(f"Cluster {i} - number of nodes: {len(cluster_nodes)}")
    top_sources = df[df['srcaddr'].isin(cluster_nodes)]['srcaddr'].value_counts().nlargest(5)
    top_targets = df[df['dstaddr'].isin(cluster_nodes)]['dstaddr'].value_counts().nlargest(5)
    print(f"Top sources:\n{top_sources}")
    print(f"Top targets:\n{top_targets}\n")


