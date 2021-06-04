#!/usr/bin/env python3
# -*- coding=utf-8 -*-


# sys
import math


class PriorQueMin(object):
    def __init__(self, vals):
        ns = len(vals)
        self.height = int(math.ceil(math.log(ns, 2)))
        self.nleaves = 2 ** self.height
        self.nbranches = self.nleaves - 1
        self.n = self.nleaves + self.nbranches
        self.nodes = [float('inf')] * self.n
        for i in range(ns):
            self.nodes[self.nbranches + i] = vals[i]
        for i in reversed(range(self.nbranches)):
            self.nodes[i] = min(self.nodes[2*i + 1], self.nodes[2*i + 2])

    def getMin(self):
        cur = 0
        for i in range(self.height):
            if self.nodes[2 * cur + 1] < self.nodes[2 * cur + 2]:
                cur = 2 * cur + 1
            else:
                cur = 2 * cur + 2
        return (cur - self.nbranches, self.nodes[cur])

    def pop(self):
        p, v = self.getMin()
        self.change(p, float('inf'))
        return (p, v)

    def change(self, idx, delta):
        cur = self.nbranches + idx
        self.nodes[cur] += delta
        for i in range(self.height):
            cur = (cur - 1) // 2
            next_parent = min(self.nodes[2*cur + 1], self.nodes[2*cur + 2])
            if self.nodes[cur] == next_parent:
                break
            self.nodes[cur] = next_parent

    def dump(self):
        print("# leaves: %d, # branches: %d, n: %d nodes" % (self.nleaves, self.nbranches, self.n))
        res = ""
        cur = 0
        for i in range(self.height + 1):
            for j in range(2 ** i):
                res += str(self.nodes[cur])
                cur += 1
            res += '\n'
        return res
