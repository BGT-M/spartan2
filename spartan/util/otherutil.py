import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class RectHistogram:
    '''
       xscale: [ 'linear' | 'log' ]
               Use a linear or log10 scale on the horizontal axis.
       yscale: [ 'linear' | 'log' ]
               Use a linear or log10 scale on the vertical axis.
       gridsize: [ 100 | integer | tuple ]
               The number of rectangles in the x-direction, default is 100. The
               corresponding number of rectangles in the y-direction is chosen such that
               the rectangles are square. Alternatively, gridsize can be
               a tuple with two elements specifying the number of rectangles in the
               x-direction and the y-direction.
       '''
    xscale = 'log'
    yscale = 'log'
    gridsize = 200

    def __init__(self, xscale='log', yscale='log', gridsize=200):
        self.xscale = xscale
        self.yscale = yscale
        self.gridsize = gridsize

    def draw(self, xs, ys, outfig=None, colorscale=True,
             suptitle='Rectangle binning points', xlabel='', ylabel=''):
        xs = np.array(xs) if type(xs) is list else xs
        ys = np.array(ys) if type(ys) is list else ys
        if self.xscale == 'log' and min(xs) <= 0:
            print('[Warning] logscale with nonpositive values in x coord')
            print('\tremove {} nonpositives'.format(len(np.argwhere(xs <= 0))))
            xg0 = xs > 0
            xs = xs[xg0]
            ys = ys[xg0]
        if self.yscale == 'log' and min(ys) <= 0:
            print('[Warning] logscale with nonpositive values in y coord')
            print('\tremove {} nonpositives'.format(len(np.argwhere(ys <= 0))))
            yg0 = ys > 0
            xs = xs[yg0]
            ys = ys[yg0]

        # color scale
        cnorm = matplotlib.colors.LogNorm() if colorscale else matplotlib.colors.Normalize()
        suptitle = suptitle + ' with a log color scale' if colorscale else suptitle

        # axis space
        if isinstance(self.gridsize, tuple):
            xgridsize = self.gridsize[0]
            ygridsize = self.gridsize[1]
        else:
            xgridsize = ygridsize = self.gridsize
        if self.xscale == 'log':
            xlogmax = np.ceil(np.log10(max(xs)))
            x_space = np.logspace(0, xlogmax, xgridsize)
        else:
            x_space = xgridsize
        if self.yscale == 'log':
            ylogmax = np.ceil(np.log10(max(ys)))
            y_space = np.logspace(0, ylogmax, ygridsize)
        else:
            y_space = ygridsize
        '''
        H: 2D array
        The bi-dimensional histogram of samples x and y. 
        xedges: 1D array
        The bin edges along the x axis.
        yedges: 1D array
        The bin edges along the y axis.
        '''
        H, xedges, yedges, fig = plt.hist2d(xs, ys, bins=(x_space, y_space), cmin=1, norm=cnorm, cmap=plt.cm.jet)
        plt.xscale(self.xscale)
        plt.yscale(self.yscale)

        plt.title(suptitle)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        cb = plt.colorbar()
        if colorscale:
            cb.set_label('log10(N)')
        else:
            cb.set_label('counts')

        if outfig is not None:
            fig.savefig(outfig)
        return fig, H, xedges, yedges

    def find_peak_rect(self, xs, ys, H, xedges, yedges, x, y, radius):
        '''
        bi-dimensional array H: the number of samples of bins
        xedges: The bin edges along the first dimension
        yedges: The bin edges along the second dimension
        return: the bin with the largest number of samples in the range of
                horizontal axis: [x-radius, x+radius]
                vertical axis: [y-radius, y+radius]
        '''
        xs = np.array(xs) if type(xs) is list else xs
        ys = np.array(ys) if type(ys) is list else ys
        if self.xscale == 'log' and min(xs) <= 0:
            print('[Warning] logscale with nonpositive values in x coord')
            print('\tremove {} nonpositives'.format(len(np.argwhere(xs <= 0))))
            xg0 = xs > 0
            xs = xs[xg0]
            ys = ys[xg0]
        if self.yscale == 'log' and min(ys) <= 0:
            print('[Warning] logscale with nonpositive values in y coord')
            print('\tremove {} nonpositives'.format(len(np.argwhere(ys <= 0))))
            yg0 = ys > 0
            xs = xs[yg0]
            ys = ys[yg0]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        '''
            x range: [x - radius, x + radius]
            y range: [y - radius, y + radius]
        '''
        if x - radius < xmin:
            xst = xmin
        else:
            xst = x - radius
        if x + radius > xmax:
            xend = xmax
        else:
            xend = x + radius
        if y - radius < ymin:
            yst = ymin
        else:
            yst = y - radius
        if y + radius > ymax:
            yend = ymax
        else:
            yend = y + radius

        'find the bins in the range'
        xstpoint = np.argwhere(xedges >= xst)[0][0]
        xendpoint = np.argwhere(xedges <= xend)[-1][0]
        xbins = range(xstpoint, xendpoint)
        ystpoint = np.argwhere(yedges >= yst)[0][0]
        yendpoint = np.argwhere(yedges <= yend)[-1][0]
        ybins = range(ystpoint, yendpoint)
        'find bin with the largest number of samples in the range'
        H1 = H[xbins]
        H2 = H1[:, ybins]
        maxpoint = np.argmax(H2)
        'the index of the largest number of array H2'
        xindex, yindex = int(maxpoint / H2.shape[1]), int(maxpoint % H2.shape[1])
        'the index of bin with maximum number'
        xbinid, ybinid = xindex + xstpoint, yindex + ystpoint

        'the range of max bin'
        binxst, binxend, binyst, binyend = xedges[xbinid], xedges[xbinid + 1], \
                                           yedges[ybinid], yedges[ybinid + 1]
        'return coordinate pairs in the max bin'
        xcoords = set(np.argwhere(xs >= binxst).T[0]) & set(np.argwhere(xs <= binxend).T[0])
        ycoords = set(np.argwhere(ys >= binyst).T[0]) & set(np.argwhere(ys <= binyend).T[0])
        pairids = list(xcoords & ycoords)
        coordpairs = list(zip(xs[pairids], ys[pairids]))
        return coordpairs