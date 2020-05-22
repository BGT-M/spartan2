import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def drawScatterPoints(xs, ys, outfig=None, suptitle="scatter points",
                     xlabel='x', ylabel='y'):
    fig = plt.figure()
    fig.suptitle(suptitle)
    plt.scatter(xs, ys, marker='.')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if outfig is not None:
        fig.savefig(outfig)
    return fig

def drawHexbin(xs, ys, outfig=None, xscale = 'log', yscale= 'log',
               gridsize = 200,
               suptitle='Hexagon binning points',
               colorscale=True ):
    '''
        xscale: [ 'linear' | 'log' ]
            Use a linear or log10 scale on the horizontal axis.
	yscale: [ 'linear' | 'log' ]
            Use a linear or log10 scale on the vertical axis.
        gridsize: [ 100 | integer ]
            The number of hexagons in the x-direction, default is 100. The
            corresponding number of hexagons in the y-direction is chosen such that
            the hexagons are approximately regular. Alternatively, gridsize can be
            a tuple with two elements specifying the number of hexagons in the
            x-direction and the y-direction.
    '''
    if xscale == 'log' and min(xs)<=0:
        print('[Warning] logscale with nonpositive values in x coord')
        print('\tremove {} nonpositives'.format(len(np.argwhere(xs<=0))))
        xg0=xs>0
        xs = xs[xg0]
        ys = ys[xg0]
    if yscale == 'log' and min(ys)<=0:
        print('[Warning] logscale with nonpositive values in y coord')
        print( '\tremove {} nonpositives'.format(len(np.argwhere(ys<=0))))
        yg0=ys>0
        xs = xs[yg0]
        ys = ys[yg0]

    fig = plt.figure()
    if colorscale:
        plt.hexbin(xs, ys, bins='log', gridsize=gridsize, xscale=xscale,
                   yscale=yscale, mincnt=1, cmap=plt.cm.jet)
        plt.title(suptitle+' with a log color scale')
        cb = plt.colorbar()
        cb.set_label('log10(N)')
    else:
        plt.hexbin(xs, ys, gridsize=gridsize, xscale=xscale, yscale=yscale,
                   mincnt=1, cmap=plt.cm.jet)
        plt.title(suptitle)
        cb = plt.colorbar()
        cb.set_label('counts')
    #plt.axis([xmin, xmax, ymin, ymax])
    if outfig is not None:
        fig.savefig(outfig)
    return fig

def drawRectbin(xs, ys, outfig=None, xscale = 'log', yscale= 'log',
               gridsize = 200,
               suptitle='Rectangle binning points',
               colorscale=True, xlabel='', ylabel=''):
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
    xs = np.array(xs) if type(xs) is list else xs
    ys = np.array(ys) if type(ys) is list else ys
    if xscale == 'log' and min(xs)<=0:
        print('[Warning] logscale with nonpositive values in x coord')
        print('\tremove {} nonpositives'.format(len(np.argwhere(xs<=0))))
        xg0=xs>0
        xs = xs[xg0]
        ys = ys[xg0]
    if yscale == 'log' and min(ys)<=0:
        print( '[Warning] logscale with nonpositive values in y coord' )
        print( '\tremove {} nonpositives'.format(len(np.argwhere(ys<=0))) )
        yg0=ys>0
        xs = xs[yg0]
        ys = ys[yg0]

    fig = plt.figure()

    # color scale
    cnorm = matplotlib.colors.LogNorm() if colorscale else matplotlib.colors.Normalize()
    suptitle = suptitle+' with a log color scale' if colorscale else \
            suptitle

    # axis space
    if xscale=='log':
        xlogmax = np.ceil(np.log10(max(xs)))
        x_space = np.logspace(0, ylogmax, gridsize)
    else:
        x_space = gridsize
    if yscale=='log':
        ylogmax = np.ceil(np.log10(max(ys)))
        y_space = np.logspace(0, xlogmax, gridsize)
    else:
        y_space = gridsize

    plt.hist2d(xs, ys, bins=(x_space, y_space),  cmin=1, norm=cnorm,
            cmap=plt.cm.jet )
    plt.xscale(xscale)
    plt.yscale(yscale)

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
    return ''


def findMaxRectbin(xs, ys, x, y, radius, gridsize=100, xscale='log', yscale='log'):
    '''
    gridsize: [ 100 | integer ]
            The number of hexagons in the x-direction, default is 100. The
            corresponding number of hexagons in the y-direction is chosen such that
            the hexagons are approximately regular. Alternatively, gridsize can be
            a tuple with two elements specifying the number of hexagons in the
            x-direction and the y-direction.
    mode: [ 'linear' | 'log' ]
    return: the bin with the largest number of samples in the range of
            horizontal axis: [x-radius, x+radius]
            vertical axis: [y-radius, y+radius]
    '''
    xs = np.array(xs) if type(xs) is list else xs
    ys = np.array(ys) if type(ys) is list else ys
    if xscale == 'log' and min(xs) <= 0:
        print('[Warning] logscale with nonpositive values in x coord')
        print('\tremove {} nonpositives'.format(len(np.argwhere(xs <= 0))))
        xg0 = xs > 0
        xs = xs[xg0]
        ys = ys[xg0]
    if yscale == 'log' and min(ys) <= 0:
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
    '''
        bi-dimensional array H: the number of samples of bins
        xedges: The bin edges along the first dimension
        yedges: The bin edges along the second dimension
    '''
    H, xedges, yedges = np.histogram2d(xs, ys, gridsize)
    xpoints = list(set(np.argwhere(xedges >= xst).T[0]) & set(np.argwhere(xedges <= xend).T[0]))
    ypoints = list(set(np.argwhere(yedges >= yst).T[0]) & set(np.argwhere(yedges <= yend).T[0]))
    # xoffset, yoffset = min(xpoints), min(ypoints)
    'find bin with the largest number of samples in the range'
    H1 = H[xpoints]
    H2 = H1[:, ypoints]
    maxpoint = np.argmax(H2)
    'the index of the largest number of array H2'
    xindex, yindex = int(maxpoint / H2.shape[1]) + min(xpoints), int(maxpoint % H2.shape[1]) + min(ypoints)
    xindex, yindex = int(maxpoint / H2.shape[1]) + min(xpoints), int(maxpoint % H2.shape[1]) + min(ypoints)
    return xindex, yindex


def drawTimeseries(T, S, bins='auto', savepath='', savefn=None, dumpfn=None):
    ts = np.histogram(T,bins=bins)
    y = np.append([0],ts[0])
    f=plt.figure()
    plt.plot(ts[1], y, 'r-+')
    if len(S)>0:
        ssts = np.histogram(S, bins=ts[1])
        ssy = np.append([0], ssts[0])
        plt.plot(ssts[1], ssy, 'b-*')
    if savefn is not None:
        f.savefig(savepath+savefn)
    if dumpfn is not None:
        import pickle
        pickle.dump(f, file(savepath+dumpfn,'w'))
    return f


def userTimeSeries(lts, twindow):
    mints=min(lts)
    maxts=max(lts)
    shiftlts = [x-mints for x in lts]
    series={}
    for i in range((maxts-mints)/twindow + 1):
        series[i]=0
    for x in shiftlts:
        series[x/twindow] += 1

    xcoords=[]
    for i in range(len(series)):
        xcoords.append(mints+twindow*i+int(twindow/2.0))

    return xcoords, series
