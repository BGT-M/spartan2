import os
import sys
import numpy as np
from datetime import datetime
import time

"gzip file must be read and write in binary/bytes"
def myopenfile(fnm, mode):
    f = None
    if 'w' in mode:
        if isgzfile(fnm):
            import gzip
            mode = mode+'b' if 'b' != mode[-1] else mode
            f = gzip.open(fnm, mode)
        else:
            f = open(fnm, mode)
    else:
        if 'r' not in mode and 'a' not in mode:
            mode = 'r' + mode
        if os.path.isfile(fnm):
            if not isgzfile(fnm):
                f = open(fnm, mode)
            else:
                import gzip
                mode = 'rb'
                f = gzip.open( fnm, mode )
        elif os.path.isfile(fnm+'.gz'):
            'file @fnm does not exists, use fnm.gz instead'
            print(
                '==file {} does not exists, read {}.gz instead'.format(fnm,
                                                                       fnm))
            import gzip
            mode = 'rb'
            f = gzip.open(fnm+'.gz', mode)
        else:
            print('file: {} or its zip file does NOT exist'.format(fnm))
            sys.exit(1)
    return f

def isgzfile( filename ):
    return filename.endswith(".gz")

# delim must be str
def saveSimpleDictData(simdict, outdata, delim=':', mode='w'):
    if isgzfile(outdata):
        mode = mode + 'b' if mode[-1] != 'b' else mode
    # write bytes
    ib = True if 'b' in mode else False

    with myopenfile(outdata, mode) as fw:
        for k, val in simdict.items():
            k = k.decode() if isinstance(k, bytes) else str(k)
            val = val.decode() if isinstance(val, bytes) else str(val)
            outstr = "{}{}{}\n".format(k, delim,  val)
            x = outstr.encode() if ib else outstr
            fw.write(x)
        fw.close()

def loadSimpleDictData(indata, delim=':', mode='r', dtypes=[int, int]):
    if isgzfile(indata):
        mode = 'rb'
    # bytes
    ib = True if 'b' in mode else False

    simdict={}
    with myopenfile(indata, mode) as fr:
        for line in fr:
            line = line.decode() if ib else str(line)
            line = line.strip().split(delim)
            ktype, vtype = dtypes
            simdict[ktype(line[0])] = vtype(line[1])
        fr.close()
    return simdict

def saveDictListData(dictls, outdata, delim=':', mode='w'):
    if isgzfile(outdata):
        'possible mode is ab'
        mode = mode + 'b' if mode[-1] != 'b' else mode
    # write bytes
    ib = True if 'b' in mode else False

    with myopenfile(outdata, mode) as fw:
        i=0
        for k, l in dictls.items():
            if not isinstance(l,(list, np.ndarray)):
                print("This is not a dict of value list.", type(l))
                break
            if len(l)<1:
                continue
            k = k.decode() if isinstance(k, bytes) else str(k)
            ostr = "{}{}".format(k,delim)
            if len(l)<1:
                i += 1
                continue
            l = [ x.decode() if isinstance(x,bytes) else str(x) for x in l ]
            ostr = ostr + " ".join(l) + '\n'
            fw.write(ostr.encode() if ib else ostr)
        fw.close()
        if i > 0:
            print( "Warn: total {} empty dict lists are removed".format(str(i)) )


def loadDictListData(indata, ktype=str, vtype=str, delim=':', mode='r'):
    if isgzfile(indata):
        mode = 'rb'
    # bytes
    ib = True if 'b' in mode else False

    dictls={}
    with myopenfile(indata, mode) as fr:
        for line in fr:
            line = line.decode() if ib else line
            line = line.strip().split(delim)
            lst=[]
            for e in line[1].strip().split(' '):
                lst.append(vtype(e))
            dictls[ktype(line[0].strip())]=lst
        fr.close()
    return dictls

def convert_to_db_type(basic_type):
    basic_type_dict = {
        int: "INT",
        str: "TEXT",
        float: "REAL"
    }
    if basic_type in basic_type_dict.keys():
        return basic_type_dict[basic_type]
    else:
        return "TEXT"


def renumberids(indir, outdir, fnm, ofnm, delimeter=' ', comments='#', nodetype=str):
    users, msgs = {}, {}
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(indir, fnm)
    ofilepath = os.path.join(outdir, ofnm)
    with myopenfile(filepath, 'r') as f, myopenfile(ofilepath, 'w') as outf:
        for line in f:
            line = line.strip()
            if line.startswith(comments):
                continue
            elems = line.split(delimeter)
            uid = nodetype(elems[0])
            bid = nodetype(elems[1])
            if uid not in users:
                users[uid] = len(users)
            if bid not in msgs:
                msgs[bid] = len(msgs)
            #t = int(elems[2])
            #label = int(elems[3])
            pos = line.find(delimeter)
            pos = line.find(delimeter, pos+1)
            outf.write('{} {} {}\n'.format(
                users[uid], msgs[bid], line[pos+1:]))
        outf.close()
        f.close()

    with open(os.path.join(outdir, 'userid.dict'), 'w') as f1,\
            open(os.path.join(outdir, 'msgid.dict'), 'w') as f2:
        for u, val in users.items():
            f1.write("{},{}\n".format(u, val))
        for m, val in msgs.items():
            f2.write("{},{}\n".format(m, val))
        f1.close()
        f2.close()
    return len(users), len(msgs)


'''
  @nodetype is the assumed type of node id, which used for saving the space of
  dict keys
  hdfs default delimeter is '\x01'
  If dicts exists , then reuse this dicts, and append new kyes when necessary.
'''
def renumberids2(infiles, outdir, delimeter =' ', isbyte=False,
        comments ='#', nodetype=str, colidx =[0, 1], dicts=None):
# for python >= 3.7
#def renumberids2(infiles, outdir, delimeter:str =' ', isbyte=False,
#        comments:str='#', nodetype=str, colidx:list =[0, 1], dicts:list=None):
    mode = 'b' if isbyte else ''
    numids = len(colidx)
    nodes = []
    if dicts is  None:
        nodes = [ {} for i in range(numids) ]
    elif len(dicts)==numids:
        for i in range(numids):
            if type(dicts[i]) is str:
                nodes.append(loadSimpleDictData(dicts[i], mode=mode,
                        dtypes=[nodetype, int]))
            elif type(dicts[i]) is dict:
                nodes.append(dicts[i])
            else:
                print("Error:\tincorrect type of dicts element: str or dict.")
                print("\tuse empty dict instead")
                nodes.append({})
    else:
        print("Error: incorrect input of dicts, {}".format(dicts))
        sys.exit(1)

    delimeter = delimeter.decode() if type(delimeter) is bytes else \
            str(delimeter)
    comments = comments.decode() if type(comments) is bytes else \
            str(comments)

    #=users, msgs = {}, {}
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    #import ipdb; ipdb.set_trace()
    import glob
    files = glob.glob(infiles)
    for filepath in files:
        fnm = os.path.basename(filepath)
        #print('\tprocessing file {}'.format(fnm), flush=True)
        print('\tprocessing file {}'.format(fnm))
        sys.stdout.flush()
        j = fnm.find('.')
        nfnm = fnm[:j]+'.reid'+fnm[j:] if j!=-1 else fnm+'.reid'
        ofilepath = os.path.join(outdir, nfnm)
        with myopenfile(filepath, 'r'+mode) as f, myopenfile(ofilepath,
                                                             'w'+mode) as outf:
            for line in f:
                line = line.decode() if 'b' in mode else line
                line = line.strip()
                if line.startswith(comments):
                    continue
                elems = line.split(delimeter)
                nelems = elems.copy()
                for i in range(numids):
                    nodedict = nodes[i] # e.g. users
                    elemidx = colidx[i]
                    orgid = elems[elemidx].encode() if nodetype is bytes \
                            else nodetype(elems[elemidx])
                    if orgid not in nodedict:
                        nodedict[orgid] = len(nodedict)
                    nelems[elemidx] = str(nodedict[orgid])
                outstr = delimeter.join(nelems) + '\n'
                outx = outstr.encode() if 'b' in mode else outstr
                outf.write(outx)
            outf.close()
            f.close()

    for i in range(numids):
        nodedict = nodes[i]
        elemidx = colidx[i]
        outnm = os.path.join(outdir, "col{}ids.dict".format(elemidx))
        saveSimpleDictData(nodedict, outnm, delim=':', mode='b')
    return [ len(nodes[i]) for i in range(numids) ]


'''
    extract time stamps in log files or edgelist tensor
    @groupids the group col idx used for aggregating timestamps
    '\x01'
'''
def extracttimes(infile, outfile, timeidx=0, timeformat='%Y-%m-%d %H:%M:%S', delimeter=' ',
        isbyte=False, comments='#', nodetype=str, groupids=[]):
    mode = 'b' if isbyte else ''
    aggts = {} # final dict list for aggregating time series.
    import glob
    files = glob.glob(infile)
    for filepath in files:
        fnm = os.path.basename(filepath)
        #print('\tprocessing file {}'.format(fnm), flush=True)
        print('\tprocessing file {}'.format(fnm))
        sys.stdout.flush()
        with myopenfile(filepath, 'r'+mode) as f:
            for line in f:
                line = line.decode() if isinstance(line, bytes) else line
                if line.startswith(comments):
                    continue
                elems = line.split(delimeter)
                # todo: convert time string to ts
                if timeformat != 'int':
                    date = datetime.strptime(elems[timeidx], timeformat)
                    ts = int(time.mktime(date.timetuple()))
                # group by groupid
                if len(groupids) == 1:
                    key = elems[groupids[0]]
                else:
                    key = ','.join(np.array(elems)[groupids])
                if key not in aggts:
                    aggts[key] = []
                aggts[key].append(ts)
    if outfile is not None:
        saveDictListData(aggts, outfile)
    return aggts

'''
time\x01uid\x01...
e.g.: uid:t1,t2,t3
e.g.: uid,msg:t1,t2,t3
'''
