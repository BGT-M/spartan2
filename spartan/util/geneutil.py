
from gensim import similarities
import os, logging
import numpy as np
import pandas as pd

class NeighborSampler():
    def __init__(self, chip_name, stage = 0, x_data=None):
        self.index = None
        self.chip_name = chip_name
        self.stage = stage
        self.build(x_data)

    def build(self, x_data=np.array([])):
        corpus = []
        for i in range(x_data.shape[0]):
            gensim_format_vec = []
            for j in range(x_data.shape[1]):
                gensim_format_vec.append((j, x_data[i][j]))
            corpus.append(gensim_format_vec)
        logging.info("#sample to build index: %s" % x_data.shape[0])
        self.get_index(corpus,x_data.shape[1])

    def get_index(self, corpus=None, n_feature=None):
        self.index = similarities.Similarity("%s_%s_neighbor.index.tmp" % (self.chip_name, self.stage), corpus, num_features=n_feature)
        return self.index

    def get_topk(self, vec, k):
        gensim_format_vec = []
        for i in range(len(vec)):
            gensim_format_vec.append((i, vec[i]))
        sims = self.index[gensim_format_vec]

        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1])
        top_k = sim_sort[0:k]
        return top_k

def build_sampled_coexpression_matrix(x_data, filename, k = 1000):
    logging.info("Co-expression matrix building process starts.")
    logging.info("data shape: %s %s" % (x_data.shape[0], x_data.shape[1]))
    n_gene = x_data.shape[1]

    neighbor_sampler = NeighborSampler(filename, x_data = x_data)
    with open("matrix.txt", "w") as f:
        for i in range(n_gene):
            list_neighbor = neighbor_sampler.get_topk(x_data[i], k)
            list_neighbor.extend(neighbor_sampler.get_topk(x_data[i], k))
            list_neighbor = list(set(list_neighbor))
            logging.info("sample %s's topk neighbor: %s" % (i, list_neighbor))
            for j, value in list_neighbor:
                pearson_value = pearson(x_data[i], x_data[j])
                f.write("%s\t%s\t%s\n" % (i, j, pearson_value))
            f.flush()
            os.fsync(f.fileno())


def pearson(X, Y):
    return np.corrcoef(X, Y)[0][1]

if __name__ == "__main__":
    x_data = pd.read_csv("F:\gene2\spartan2\live-tutorials\inputData\example.csv", sep=",", header=None)
    build_sampled_coexpression_matrix(x_data, "out")