from gensim import similarities
import os, logging
import numpy as np
import pandas as pd
import gc

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
        self.index = similarities.MatrixSimilarity(corpus, num_features=n_feature)
        return self.index

    def get_topk(self, vec, k):
        gensim_format_vec = []
        for i in range(len(vec)):
            gensim_format_vec.append((i, vec[i]))
        sims = self.index[gensim_format_vec]

        sim_sort = sorted(list(enumerate(sims)), key=lambda item: item[1])
        top_k = sim_sort[0:k]
        return top_k


def load_gene_file(filapath, stage_index, output_filepath_list):
    list_data = []
    with open(filapath) as fp:
        next(fp) # skip header
        for line in fp:
            data = line.split("\t")[1:]
            x_data = np.zeros(len(data))
            for i in range(len(data)):
                x_data[i] = float(data[i])
            list_data.append(x_data)
            del data
            gc.collect() # the line has more than 800K element
    x_data = np.array(list_data)

    if stage_index:
        print(stage_index)
        # stage_index = stage_index.split(",")
        stage_index = [int(x) for x in stage_index]
        stage_index = np.array(stage_index)
    else:
        stage_index = np.array([0] * x_data.shape[0])
    n_stage = np.max(stage_index) + 1
    # print("gene file has " + str(n_stage) + " stages")
    for j in range(n_stage):
        idx = stage_index == j
        x_output = x_data[idx]
        np.savetxt(output_filepath_list[j], x_output, delimiter="\t")

def build_sampled_coexpression_matrix(x_data, argsoutput1, k = 1000, use_peason = False):
    logging.info("Co-expression matrix building process starts.")
    logging.info("data shape: %s %s" % (x_data.shape[0], x_data.shape[1]))
    n_gene = x_data.shape[0]

    neighbor_sampler = NeighborSampler("out", x_data = x_data)
    with open(argsoutput1, "w") as f:
        for i in range(n_gene):
            list_neighbor = neighbor_sampler.get_topk(x_data[i], k)
            list_neighbor.extend(neighbor_sampler.get_topk(x_data[i], k))
            list_neighbor = list(set(list_neighbor))
            logging.info("sample %s's topk neighbor: %s" % (i, list_neighbor))
            for j, value in list_neighbor:
                if use_peason:
                    value = pearson(x_data[i], x_data[j])
                f.write("%s\t%s\t%s\n" % (i, j, value))
            f.flush()
            os.fsync(f.fileno())

def get_top_by_threshold(input_path, output_path, threshold = 0.99):
    logging.info("start getting top by threshold")

    with open(output_path, "w") as wfp:
        with open(input_path) as fp:
            for line in fp:
                tokens = line.split()
                if len(tokens) != 3:
                    continue
                if threshold <= float(tokens[2]):
                    wfp.write("%s\t%s\n" % (tokens[0], tokens[1]))

def pearson(X, Y):
    return np.corrcoef(X, Y)[0][1]

def graph1_minus_graph2(g1, g2, only_pos = 0, output_path = "res_net.edgelist"):
    """
    If want to get edges in g1 and not in g2, only_pos = -1
    If want to get edges in g2 and not in g1, only_pos = 1
    """
    edge_list = []
    nodes1 = g1.nodes()
    nodes2 = set(g2.nodes())
    for node in nodes1:
        neighbors2 = set()
        neighbors1 = set(g1.neighbors(node))
        if node in nodes2:
            neighbors2 = set(g2.neighbors(node))

    if only_pos != 1:
        for neighbor in neighbors1:
            if neighbor not in neighbors2:
                edge_list.append((node, neighbor))
    if only_pos != -1:
        for neighbor in neighbors2:
            if neighbor not in neighbors1:
                edge_list.append((node, neighbor))

    with open(output_path, "w") as wfp:
        for edge in edge_list:
            wfp.write(str(edge[0])+"\t"+str(edge[1])+"\n")
    return edge_list

if __name__ == "__main__":
    x_data = pd.read_csv("F:\gene2\spartan2\live-tutorials\inputData\example.csv", sep=",", header=None)
    build_sampled_coexpression_matrix(x_data, "out")