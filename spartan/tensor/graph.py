#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   graph.py
@Desc    :   Definition of graph structure.
'''

# here put the import lib


class Graph:
    def __init__(self, value_tensor, attr_tensor):
        self.attr_tensor = attr_tensor
        self.value_tensor = self.handle_value(value_tensor)

    def handle_value(self, value_tensor):
        if value_tensor is None:
            return None
        else:
            return value_tensor
