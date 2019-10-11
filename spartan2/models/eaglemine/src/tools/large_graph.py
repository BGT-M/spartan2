#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

#  Project: eaglemine
#     Extract node features for very large graph.
#     To use this script, the GraphLab for very large graph processing.
#        GraphLab:  is a graph-based, high performance, distributed computation framework written in C++.
#           The GraphLab project was started by Prof. Carlos Guestrin of Carnegie Mellon University in 2009.
#           It is an open source project using an Apache License. While GraphLab was originally developed for
#           Machine Learning tasks, it has found great success at a broad range of other data-mining tasks;
#           out-performing other abstractions by orders of magnitude.      ---- from Wikipedia: GraphLab
#
#           Reference to:  https://turi.com/download/install-graphlab-create.html
#
#    large_graph.py
#      Version:  1.0
#      Goal: Subroutine script
#      Created by @wenchieh  on <12/25/2017>
#

__author__ = 'wenchieh'

# sys
import time

# third-party lib
import numpy as np
import graphlab as gl


def large_graph_features(users_sf, rels_sf, outfn, nodename, verbose=False):
    tol_sg = gl.SGraph()
    tol_sg = tol_sg.add_vertices(vertices=users_sf, vid_field=nodename)
    tol_sg = tol_sg.add_edges(edges=rels_sf, src_field='src', dst_field='dst')

    total_gfeat = users_sf[[nodename]]
    outputStr = '#rows: {}'.format(len(rels_sf))

    ## PageRank Feature
    print("PageRank ...")
    timePoint = time.time()
    data_m = gl.pagerank.create(tol_sg, verbose=False)
    outputStr += "Pagerank, "+str(time.time()-timePoint)+"\n"
    tempfeat_sf = data_m['pagerank']
    tempfeat_sf.rename({'__id':nodename})
    total_gfeat = total_gfeat.join(tempfeat_sf, on=nodename, how='left')
    total_gfeat.remove_column('delta')

    ## Triangle Count Feature
    print("Triangle Count ...")
    timePoint = time.time()
    data_m = gl.triangle_counting.create(tol_sg, verbose=False)
    outputStr += "Triangle_Count, "+str(time.time()-timePoint)+"\n"
    tempfeat_sf = data_m['triangle_count']
    tempfeat_sf.rename({'__id':nodename})
    total_gfeat = total_gfeat.join(tempfeat_sf, on=nodename, how='left')

    ## K-core Feature
    print("k-core ...")
    timePoint = time.time()
    data_m = gl.kcore.create(tol_sg, verbose=False)
    outputStr += "k-core, "+str(time.time()-timePoint)+"\n"
    tempfeat_sf = data_m['core_id']
    tempfeat_sf.rename({'__id':nodename})
    total_gfeat = total_gfeat.join(tempfeat_sf, on=nodename, how='left')

    ## Out-degree Feature
    print("Out-degree ...")
    timePoint = time.time()
    tempfeat_sf = rels_sf.groupby("src", {'out_degree':gl.aggregate.COUNT()})
    outputStr += "out_degree, "+str(time.time()-timePoint)+"\n"
    tempfeat_sf.rename({'src':nodename})
    total_gfeat = total_gfeat.join(tempfeat_sf, on=nodename, how='left')

    ## In-degree Feature
    print("In-degree ...")
    timePoint = time.time()
    tempfeat_sf = rels_sf.groupby("dst", {'in_degree':gl.aggregate.COUNT()})
    outputStr += "in_degree, "+str(time.time()-timePoint)+"\n"
    tempfeat_sf.rename({'dst':nodename})
    total_gfeat = total_gfeat.join(tempfeat_sf, on=nodename, how='left')

    ## Filling the missing values
    total_gfeat = total_gfeat.fillna('pagerank',0)
    total_gfeat = total_gfeat.fillna('triangle_count',0)
    total_gfeat = total_gfeat.fillna('core_id',0)
    total_gfeat = total_gfeat.fillna('out_degree',0)
    total_gfeat = total_gfeat.fillna('in_degree',0)

    total_gfeat['total_degree'] = np.array(total_gfeat['out_degree']) + \
                                  np.array(total_gfeat['in_degree'])
    total_gfeat = total_gfeat.sort(nodename)
    total_gfeat.save(outfn, format='csv')
    print("done!")
    if verbose:  print("Logging: " + outputStr)


if __name__ == '__main__':
    inpath = '../example/'
    relations_infn = 'example.graph'
    outpath = '../output/'
    outfn = 'graph_node_feature.csv'
    rels_sf = gl.SFrame.read_csv(inpath + relations_infn, header=False, delimiter=',', comment_char='#')
    rels_sf.rename({'X1':'src','X2':'sink'})

    uids = np.unique(rels_sf.to_numpy())
    users_sf = gl.SFrame({"nodeId": uids})
    large_graph_features(users_sf, rels_sf, outpath + outfn, 'nodeId')