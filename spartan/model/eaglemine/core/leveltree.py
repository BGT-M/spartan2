#!/usr/bin/python2.7
# -*- coding=utf-8 -*-

# #  WaterLevel tree class for identifying micro-clusters / islands in
#    multi-dimensional (>=2) histogram
#  Author: wenchieh
#
#  Project: eaglemine
#      leveltree.py
#      Version:  1.0
#      Date: November 27 2017
#      Main Contact: Wenchieh Feng (wenchiehfeng.us@gmail.com)
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <11/27/2017>


__author__ = 'wenchieh'

# sys
import operator
from itertools import product
from collections import deque

# third-party lib
import numpy as np
import scipy.ndimage.morphology as snm


class LevelTree(object):
    class Node(object):
        _DELIMITER_ = ';'

        def __init__(self, id, level, hcubes, prnt_id, child=None):
            self.id = id
            self.level = level
            self.hcubes = hcubes
            self.prnt_id = prnt_id
            self.child = child
            ## the number of pts contained in its hypercubes
            # self.mass = None

        def get_volume(self):
            return len(self.hcubes)

        def get_covers(self):
            return self.hcubes

        def __str__(self):
            # node object to string
            obj_str = "{}{}{}{}{}".format(self.id, self._DELIMITER_, self.level, self._DELIMITER_, self.prnt_id)
            childs_str = "0" if self.child is None else str(self.child).replace(', ', ',')[1:-1]
            content = ""
            for hcb in self.hcubes:
                content += str(hcb) + self._DELIMITER_
            return obj_str + self._DELIMITER_ + childs_str + self._DELIMITER_ + content[:-1]

        @staticmethod
        def load(node_str):
            if node_str is None or len(node_str) <= 0:
                return None
            tokens = node_str.strip().split(LevelTree.Node._DELIMITER_)
            node_id, level, prnt_id = int(tokens[0]), float(tokens[1]), int(tokens[2])
            child = None if tokens[3] == "0" else list(map(int, tokens[3].split(',')))
            hcubes = [eval(tokens[i]) for i in range(4, len(tokens))]

            node = LevelTree.Node(node_id, level, hcubes, prnt_id, child)
            return node

    def __init__(self):
        self.comp_tree = dict()
        self.levels = None
        self.bound = None
        self._earth = None
        self._value = None
        self._Nnnz = 0
        self.mode = 0

    #TODO optimize the search process with [scipy.ndimage.label]
    def _search_components_(self, min_level=1.0, morphology=False, sqrprobe_size=2):
        coord2val_dict = dict()
        if morphology:
            hypr_canvas = np.zeros(self.bound)
            for k in range(self._Nnnz):
                hypr_canvas[tuple(self._earth[k])] += self._value[k]
            probe = np.ones(tuple([sqrprobe_size] * hypr_canvas.ndim))
            coords = np.nonzero(snm.binary_opening(hypr_canvas, structure=probe).astype(int))
            for pos in zip(*coords):
                coord2val_dict[pos] = hypr_canvas[pos]
        else:
            coord2val_dict = dict(zip(map(tuple, self._earth), self._value))

        connect_comps = list()
        coords = list(coord2val_dict.keys())
        while len(coords) > 0:
            pos = coords.pop()
            if coord2val_dict[pos] < min_level: continue

            component = list()
            queue = deque()
            queue.append(pos)
            while len(queue) > 0:
                cur_pos = queue.popleft()
                component.append(cur_pos)
                if cur_pos in coords:
                    coords.remove(cur_pos)

                coord_nbr_rngs = list()
                for mod in range(len(cur_pos)):
                    low, up = np.max([cur_pos[mod] - 1, 0]), np.min([cur_pos[mod] + 1, self.bound[mod]])
                    if low == up:
                        coord_nbr_rngs.append([low])
                    else:
                        coord_nbr_rngs.append(range(low, up + 1))

                for nbr in product(*coord_nbr_rngs):
                    if (pos != nbr) and (nbr in coords) and (coord2val_dict[nbr] >= min_level):
                        if nbr not in queue:
                            queue.append(nbr)
            if len(component) > 0:
                connect_comps.append(component)

        return connect_comps

    def get_nodesid_atlevel(self, level):
        nodesid = list()
        for id in self.comp_tree.keys():
            node = self.comp_tree[id]
            if node.level == level:
                nodesid.append(node.id)
            if node.level > level:
                break
        return nodesid

    def get_node(self, node_id):
        node = None
        if node_id in self.comp_tree.keys():
            node = self.comp_tree.get(node_id)
        return node

    def get_nodes(self):
        return self.comp_tree.keys()

    def get_leaves(self):
        leaves = []
        for id, node in self.comp_tree.items():
            if node.child is None:
                leaves.append(id)
        return leaves

    def build_level_tree(self, nnz_indices, nnz_elems, min_level, max_level=None, step=1.0,
                         min_hcubes=4, morphology=True, verbose=False, outfn=None):
        self.levels = list()
        self.bound = list(map(int, np.max(nnz_indices, axis=0) + 1))
        self._earth = np.asarray(nnz_indices, int)
        self._value = np.asarray(nnz_elems)
        self._Nnnz = len(nnz_indices)
        self.mode = len(self.bound)

        tiny_blobs = []
        unique_values = np.unique(self._value)
        pos2val = dict(zip(map(tuple, nnz_indices), self._value))

        if (max_level is None) or (max_level < min_level) or (max_level > np.max(self._value)):
            max_level = np.max(self._value)

        nodeid = 0
        for level in np.arange(min_level, max_level, step):
            if (level > min_level) and (2**level - 2**(self.levels[-1]) < 1.0):
                continue

            if level > min_level:
                inter_npts = np.sum((unique_values <= level) & (unique_values > self.levels[-1]))
                if inter_npts <= 0:
                    continue

            connect_comps = self._search_components_(level, morphology)
            Ncomps = len(connect_comps)
            if Ncomps == 0 or (Ncomps == 1 and len(connect_comps[0]) < min_hcubes):
                if verbose:
                    print("Info: maximum tree level: {}".format(level - step))
                break

            inserted = False
            prnt_id = -1
            if level == min_level:
                for comp in connect_comps:
                    if len(comp) <= min_hcubes:
                        tiny_blobs.extend(comp)
                        if verbose:
                            for p in comp:
                                print("{}:{}".format(p, int(2**pos2val[p]-1)))
                        continue

                    inserted = True
                    node = self.Node(nodeid, level, comp, prnt_id)
                    self.comp_tree[nodeid] = node
                    nodeid += 1
            else:
                uplevel_ids = self.get_nodesid_atlevel(self.levels[-1])
                for comp in connect_comps:
                    if len(comp) <= min_hcubes:
                        tiny_blobs.extend(comp)
                        # if verbose:
                        #     for p in comp:
                        #         print("{}:{}".format(p, int(2**pos2val[p]-1)))
                        continue
                    if len(uplevel_ids) > 1:
                        shares = [len(set(comp) & set(self.comp_tree[pid].hcubes)) for pid in uplevel_ids]
                        prnt_id = uplevel_ids[np.argmax(shares)]
                    else:
                        prnt_id = uplevel_ids[0]

                    inserted = True
                    node = self.Node(nodeid, level, comp, prnt_id)
                    self.comp_tree[nodeid] = node
                    if self.comp_tree[prnt_id].child is None:
                        self.comp_tree[prnt_id].child = list()
                    self.comp_tree[prnt_id].child.append(nodeid)
                    nodeid += 1

            if inserted is True:
                self.levels.append(level)
            else:
                if verbose:
                    print("Info: maximum tree level: {}".format(level - step))
                break

        if outfn is not None:
            if len(tiny_blobs) > 0:
                blob2cnt = dict()
                for blob in tiny_blobs:
                    blob2cnt[blob] = int(2**pos2val[blob]) - 1
                blob_pos2cnt = np.zeros((len(blob2cnt), self.mode + 1),int)
                blob_pos2cnt[:, :self.mode] = np.asarray(list(blob2cnt.keys()), int)
                blob_pos2cnt[:, self.mode] = np.asarray(list(blob2cnt.values()), int)
                np.savetxt(outfn, blob_pos2cnt, '%d', ',')

        if verbose is True:
            print("Info: Level-Tree build done!")

    def dump(self, verbose=True):
        print("Level tree basic information:")
        print("#node: {}, #level: {}".format(len(self.comp_tree), len(self.levels)))
        print("(node id): {@level, parent node, #childs, #elements}")
        nodesid = list(sorted(self.comp_tree.keys(), reverse=True))

        while len(nodesid) > 0:
            stack = deque()
            stack.append(nodesid.pop())
            while len(stack) > 0:
                node_id = stack.pop()
                if node_id in nodesid:
                    nodesid.remove(node_id)
                node = self.comp_tree.get(node_id)
                level_idx = self.levels.index(node.level)
                info = "{}, {}, {}, {}".format(node.level, node.prnt_id, node.child, node.get_volume())
                print("    " * level_idx + "|---- ({}): {} |-".format(node.id, info))
                if node.child is not None:
                    stack.extend(sorted(node.child, reverse=True))

        if verbose:
            print("dump done!")

    def __str__(self):
        infns = "Level-ree info:\n"
        infns += "mode:{}, #level:{}, #hyper-cube:{}\n".format(self.mode, len(self.levels), self._Nnnz)
        infns += "(multi-)histogram bound: {} \n".format(self.bound)
        infns += "tree levels: " + ','.join(map(str, ["%.1f"%lv for lv in self.levels])) + "\n"
        infns += "#node:{}\n".format(len(self.comp_tree))
        infns += "\t id, @level, parent, child, #element\n"
        for id, node in self.comp_tree.items():
            infns += "\t {}, {}, {}, {}, {}\n".format(node.id, node.level, node.prnt_id, node.child, node.get_volume())

        return infns

    def tree_contract(self, verbose=False):
        # level-tree nodes contract
        queue_nodesid = deque(sorted(self.comp_tree.keys(), reverse=True))
        while len(queue_nodesid) > 0:
            cur_id = queue_nodesid.popleft()
            node = self.comp_tree.get(cur_id)
            prnt_id = node.prnt_id
            if prnt_id != -1:  # tree node collapse for internal nodes
                if len(self.comp_tree.get(prnt_id).child) <= 1:
                    self.comp_tree[prnt_id].child = node.child
                    if node.child is not None:
                        for child_id in node.child:
                            self.comp_tree[child_id].prnt_id = prnt_id
                    del self.comp_tree[cur_id]

        # reserve remain levels
        remain_levels = set()
        for id, node in self.comp_tree.items():
            remain_levels.add(node.level)
        self.levels = sorted(list(remain_levels))

        if verbose:
            print("Info: Level-tree contract done!")

    def tree_prune(self, alpha=0.5, verbose=False):
        # level-tree prune and smooth tiny nodes
        pos2val = dict(zip(map(tuple, self._earth), self._value))
        node2wts = dict()

        nodesid = deque(self.get_nodesid_atlevel(self.levels[0]))
        while len(nodesid) > 0:
            node = self.comp_tree[nodesid.popleft()]
            if node.child is not None:
                # N_nd_hcbs = node.get_volume()
                if node.id not in node2wts:
                    node2wts[node.id] = np.sum([2**pos2val.get(tuple(p)) for p in node.get_covers()])
                # N_child_hcbs = 0
                N_chwts = 0
                for child_id in node.child:
                    if child_id not in node2wts:
                        node2wts[child_id] = np.sum([2**pos2val.get(tuple(p))
                                                     for p in self.comp_tree[child_id].get_covers()])
                    N_chwts += node2wts.get(child_id)
                    # N_child_hcbs += self.comp_tree[child_id].get_volume()
                # if N_child_hcbs < alpha * N_nd_hcbs:
                if N_chwts <  alpha * node2wts.get(node.id):
                    self._delete_node_(node.id)
                else:
                    nodesid.extend(node.child)

        if verbose is True:
            print("Info: Level-tree prune done!")

    def _delete_node_(self, node_id):
        node = self.comp_tree.get(node_id)
        del_childids = node.child
        if del_childids is not None:
            while len(del_childids):
                del_node = self.comp_tree.get(del_childids.pop())
                if del_node.child is not None:
                    del_childids.extend(del_node.child)
                del self.comp_tree[del_node.id]
        node.child = None

    def _get_child_ids_(self, prnt_id):
        child_ids = list()
        if prnt_id == -1:
            child_ids = self.get_nodesid_atlevel(self.levels[0])
        elif prnt_id in self.comp_tree.keys():
            child_ids = self.comp_tree.get(prnt_id).child
        return child_ids

    def _get_node_margins_(self, hcube, covers):
        hcube_margins = set()
        coord_rngnbrs = []
        for mod in range(len(hcube)):
            low, up = np.max([hcube[mod] - 1, 0]), np.min([hcube[mod] + 1, self.bound[mod]])
            if low == up:
                coord_rngnbrs.append([low])
            else:
                coord_rngnbrs.append(range(low, up+1))

        for neighbor_hcb in product(*coord_rngnbrs):
            if neighbor_hcb != hcube:
                non_neg = np.sum(neighbor_hcb) == np.sum(np.abs(neighbor_hcb))  # valid hyper-cube (non-negative pos)
                if non_neg and (neighbor_hcb not in covers):
                    hcube_margins.add(neighbor_hcb)
        return hcube_margins

    def _get_comp_margins(self, node_id):
        comp_margins = set()
        if node_id in self.comp_tree.keys():
            node = self.get_node(node_id)
            for cb in node.hcubes:
                hcube_margins = self._get_node_margins_(cb, node.hcubes)
                if len(hcube_margins) > 0:
                    comp_margins = comp_margins.union(hcube_margins)
        return comp_margins

    def get_heavynodes(self):
        heavynodes2level = dict()

        parent = -1
        while True:
            node_ids = self._get_child_ids_(parent)
            volumes = [self.get_node(id).get_volume() for id in node_ids]
            max_id = node_ids[np.argmax(volumes)]
            max_node = self.get_node(max_id)
            heavynodes2level[max_id] = max_node.level
            if max_node.child is None:
                break
            parent = max_id
        return heavynodes2level

    def tree_node_expand(self, verbose=False):
        # expand islands over same levels simultaneously
        # scope: peripheral(self-uncovered) but parent-covered cells layer-diffusing.
        # expanding order: from smallest to largest (island size), top to bottom;
        # stop criteria:
        #      1. the expanded size is same as the size it was
        #      2. has overlapping among components' expending during the process.
        #      3. has no more hcubes can be included. (no update)
        expand_queue = deque()
        expand_queue.append(-1)

        while len(expand_queue) > 0:
            prnt_id = expand_queue.popleft()
            child_ids = self._get_child_ids_(prnt_id)

            if child_ids is not None:
                if prnt_id == -1:
                    remain_hcbs = set(map(tuple, self._earth))
                else:
                    prnt_covers = self.comp_tree.get(prnt_id).get_covers()
                    remain_hcbs = set(map(tuple, prnt_covers))

                child_cores = dict()
                child_size = dict()
                for nodeid in child_ids:
                    node = self.comp_tree.get(nodeid)
                    child_cores[nodeid] = set(node.get_covers())
                    child_size[nodeid] = node.get_volume()
                    remain_hcbs = remain_hcbs.difference(child_cores[nodeid])

                sorted_ids = np.asarray(sorted(child_size.items(), key=operator.itemgetter(1)))[:, 0]
                node_expands, node_margins = dict(), dict()
                tol_expands = set()
                no_overlap = True

                # non-overlap expanding
                while no_overlap:
                    updated = False
                    kexp_size = dict()
                    for nodeid in sorted_ids:
                        if (nodeid in node_expands) and  (kexp_size.get(nodeid, 0) >= 0) and  \
                                (len(node_expands[nodeid]) >= child_size[nodeid]):
                            continue
                        else:
                            # Expend current child node
                            # 1. detect all peripheral hyper cubes as the margins for the node
                            if nodeid not in node_margins:
                                node_margins[nodeid] = self._get_comp_margins(nodeid)

                            expand_hcbs = set(node_margins[nodeid]).intersection(remain_hcbs)

                            # stop criteria check: (whether overlap with other expands so no further expansion)
                            if len(expand_hcbs.intersection(tol_expands)) > 0:
                                expand_hcbs = expand_hcbs - tol_expands
                                no_overlap = False

                            if len(expand_hcbs) < kexp_size.get(nodeid, 0):
                                kexp_size[nodeid] = -1
                                continue

                            if len(expand_hcbs) > 0:
                                # update total expand set
                                tol_expands = tol_expands.union(expand_hcbs)
                                if nodeid not in node_expands:  # # haven't been expanded so far
                                    node_expands[nodeid] = set()
                                node_expands[nodeid] = node_expands[nodeid].union(expand_hcbs)
                                kexp_size[nodeid] = len(expand_hcbs)
                                # at least one node has been updated (expanded)
                                updated = True

                                if no_overlap:
                                    # only no-overlap holds, the updating for margins of current node
                                    # will happen in next iteration
                                    # update current node margins
                                    expanded_covers = node_expands[nodeid].union(child_cores[nodeid])
                                    nextiter_margins = set()
                                    for cb in expand_hcbs:
                                        nextiter_margins = nextiter_margins.union(self._get_node_margins_(cb, expanded_covers))
                                    node_margins[nodeid] = nextiter_margins.intersection(remain_hcbs)
                                    # end if can be expand
                                    # end (else) if the island can be expand further.
                    # end for (all child nodes)
                    if updated is False:  # no expansion occur in this step.
                        break
                # end while for non-overlap expanding

                # Execute update tree node with expandable covers
                for nodeid in child_ids:
                    if nodeid not in node_expands: continue
                    self.comp_tree[nodeid].hcubes = list(node_expands[nodeid].union(child_cores[nodeid]))
                    expand_queue.append(nodeid)
            # end if (expand for internal tree node)
        # end while (expand each nodes with constraints if possible)

        if verbose is True:
            print("Info: Level-tree expand done!")

    def save_leveltree(self, outfn, sep=',', verbose=True):
        with open(outfn, 'w') as ofp:
            header = '# mode, bound, levels, #non-zeros, #tree-node, earth_value, tree_nodes\n'
            ofp.writelines(header)
            ofp.writelines(str(self.mode) + '\n')
            ofp.writelines(str(self.bound)[1:-1].replace(', ', sep) + '\n')
            ofp.writelines(sep.join(['%.1f' % lv for lv in self.levels]) + '\n')
            ofp.writelines(str(self._Nnnz) + '\n')
            ofp.writelines(str(len(self.comp_tree)) + '\n')
            for k in range(self._Nnnz):
                ofp.writelines(str(list(self._earth[k]))[1:-1].strip().replace(', ', sep) +
                               sep + str(self._value[k]) + '\n')

            for nodeid, comp in self.comp_tree.items():
                ofp.writelines(str(comp) + '\n')
            ofp.close()

        if verbose:
            print("Level-tree serialization done!")

    def load_leveltree(self, infn, sep=',', verbose=True):
        with open(infn, 'r') as ifp:
            ifp.readline()     # header line
            self.mode = int(ifp.readline().strip())
            self.bound = list(map(int, ifp.readline().strip().split(sep)))
            self.levels = list(map(float, ifp.readline().strip().split(sep)))
            self._Nnnz = int(ifp.readline().strip())
            Nnodes = int(ifp.readline().strip())
            self.mode = len(self.bound)
            self._earth = np.zeros((self._Nnnz, self.mode), int)
            self._value = np.zeros(self._Nnnz, float)
            for k in range(self._Nnnz):
                tokens = ifp.readline().strip().split(sep)
                self._earth[k] = np.array(list(map(int, tokens[:self.mode])), int)
                self._value[k] = float(tokens[self.mode])

            self.comp_tree = dict()
            for n in range(Nnodes):
                node = self.Node.load(ifp.readline().strip())
                if node is not None:
                    self.comp_tree[node.id] = node
            ifp.close()

        if verbose:
            print("Level-tree load done!")