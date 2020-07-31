
class UnionFind:
    """Union-Find data structure, provides union and find operations. Implemented with path compression optimization.
    """
    def __init__(self, N):
        """Init a N-element Union-Find with ids ranging from [0, N-1].

        Args:
            N (int): number of elements
        """
        self.N = N
        self.id = list(range(N))
        self.sizes = [1] * N

    def union(self, x, y):
        """Union two disjoint sets with id x and y respectively.

        Args:
            x (int): id of the first element
            y (int): id of the second element
        """
        rootx = self._root(x)
        rooty = self._root(y)
        if rootx == rooty:
            return

        if self.sizes[rootx] >= self.sizes[y]:
            self.id[rooty] = rootx
            self.sizes[rootx] += self.sizes[rooty]
        else:
            self.id[rootx] = rooty
            self.sizes[rooty] += self.sizes[rootx]

    def find(self, x, y):
        """Return if two elements belong to the same set.

        Args:
            x (int): id of the first element
            y (int): id of the second element

        Returns:
            [bool]: True if x and y belongs to the same set.
        """
        return self._root(x) == self._root(y)

    def _root(self, x):
        if self.id[x] == x:
            return x
        else:
            self.id[x] = self._root(self.id[x])
            return self.id[x]