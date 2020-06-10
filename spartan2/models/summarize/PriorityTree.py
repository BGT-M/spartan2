import math

class PTNode:
    def __init__(self):
        self.parent = None
        self.min_at_left = True
        self.is_leaf = False
        self.min_val = None

        self.left = None
        self.right = None


class PriorityTree:
    def __init__(self, values):
        self.values = [v for v in values]
        N = len(self.values)
        self.nodes = [None] * N

        self.root = self.build(0, N)
        self.visited_time = [0] * N

    def build(self, start, end):
        if end-start == 1:
            node = PTNode()
            node.is_leaf = True
            node.min_val = self.values[start]
            self.nodes[start] = node
            return node
        else:
            node = PTNode()
            mid = (start + end) // 2
            node.left = self.build(start, mid)
            node.right = self.build(mid, end)
            node.left.parent = node
            node.right.parent = node
            if node.left.min_val > node.right.min_val:
                node.min_at_left = False
                node.min_val = node.right.min_val
            else:
                node.min_at_left = True
                node.min_val = node.left.min_val
            return node

    def get_min(self):
        return self.root.min_val

    def pop(self):
        d, idx = self.root.min_val
        self.visited_time[idx] += 1
        self.update(idx)
        return (d, idx)

    def update(self, idx, new_val=None):
        current = self.nodes[idx]
        if new_val is not None:
            self.values[idx] = new_val
            current.min_val = new_val
            self.visited_time[idx] = 0

        def lesser(idx1, idx2):
            if self.values[idx1][0] == math.inf:
                return False
            if self.values[idx2][0] == math.inf:
                return True
            if self.visited_time[idx1] > self.visited_time[idx2]:
                return False
            if self.visited_time[idx1] == self.visited_time[idx2]:
                if self.values[idx1][0] > self.values[idx2][0]:
                    return False
            return True

        while current != self.root:
            parent = current.parent
            left = parent.left
            right = parent.right
            left_val = left.min_val
            right_val = right.min_val

            dleft, left_idx = left_val
            dright, right_idx = right_val

            minAtLeft = lesser(left_idx, right_idx)
            if minAtLeft:
                parent.min_at_left = True
                parent.min_val = left_val
            else:
                parent.min_at_left = False
                parent.min_val = right_val
            current = parent