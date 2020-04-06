import os
import sys


def myopenfile(fnm, mode):
    f = None
    if 'w' in mode:
        if '.gz' == fnm[-3:]:
            import gzip
            if 'b' not in mode:
                mode += 'b'
            f = gzip.open(fnm, mode)
        else:
            f = open(fnm, mode)
    else:
        if 'r' not in mode:
            mode = 'r' + mode
        if os.path.isfile(fnm):
            if '.gz' != fnm[-3:]:
                f = open(fnm, mode)
            else:
                import gzip
                f = gzip.open(fnm, mode)
        elif os.path.isfile(fnm+'.gz'):
            'file @fnm does not exists, use fnm.gz instead'
            print(
                '==file {} does not exists, read {}.gz instead'.format(fnm,
                                                                       fnm))
            import gzip
            f = gzip.open(fnm+'.gz', mode)
        else:
            print('file: {} or its zip file does NOT exist'.format(fnm))
            sys.exit(1)
    return f


def checkfilegz(name):
    if os.path.isfile(name):
        return name
    elif os.path.isfile(name+'.gz'):
        return name+'.gz'
    else:
        return None


def get_sep_of_file(infn):
    '''
    return the separator of the line.
    :param infn: input file
    '''
    sep = None
    with open(infn, 'r') as fp:
        for line in fp:
            if (line.startswith("%") or line.startswith("#")):
                continue
            line = line.strip()
            if (" " in line):
                sep = " "
            if ("," in line):
                sep = ","
            if (";" in line):
                sep = ';'
            if ("\t" in line):
                sep = "\t"
            break
    return sep


def saveSimpleDictData(simdict, outdata, delim=':', mode=''):
    delim = delim.decode() if type(delim) is bytes else str(delim)
    with myopenfile(outdata, 'w'+mode) as fw:
        for k, val in simdict.items():
            k = k.decode() if type(k) is bytes else str(u)
            val = val.decode() if type(val) is bytes else str(val)
            outstr = "{}{}{}\n".format(k, delim,  val)
            x = outstr.encode() if 'b' in mode else outstr
            fw.write(x)
        fw.close()

def loadSimpleDictData(indata, delim=':', mode='b', dtypes=[int, int]):
    simdict={}
    with myopenfile(indata, 'r'+mode) as fr:
        lines=fr.readlines()
        for line in lines:
            line = line.decode() if type(line) is bytes else str(line)
            line = line.strip().split(delim)
            ktype, vtype = dtypes
            simdict[ktype(line[0])] = vtype(line[1])
        fr.close()
    return simdict

def saveDictListData(dictls, outdata, delim=':'):
    with myopenfile(outdata, 'w') as fw:
        for k, l in dictls.iteritems():
            if type(l) != list:
                print("This is not a dict of value list.")
                break
            fw.write("{}{}".format(k,delim))
            for i in range(len(l)-1):
                fw.write("{} ".format(l[i]))
            fw.write("{}\n".format(l[-1]))
        fw.close()


def loadDictListData(indata, ktype=str, vtype=str):
    dictls={}
    with myopenfile(indata, 'rb') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split(':')
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
def renumberids2(infiles, outdir, delimeter:str=' ', isbyte=False,
        comments:str='#', nodetype=str, colidx:list =[0, 1], dicts:list=None):
    mode = 'b' if isbyte else ''
    numids = len(colidx)
    if dicts is  None:
        nodes = [ {} for i in range(numids) ]
    elif len(dicts)==numids:
        for i in range(numids):
            if type(dicts[i]) is str:
                nodes[i] = loadSimpleDictData(dicts[i], mode=mode,
                        dtypes=[nodetype, int])
            elif type(dicts[i]) is dict:
                nodes[i] = dicts[i]
            else:
                print("Error: incorrect type of dicts element: str or dict.")
                sys.exit(1)
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
    import glob
    files = glob.glob(infiles)
    for filepath in files:
        fnm = os.path.basename(filepath)
        print('\tprocessing file {}'.format(fnm), flush=True)
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
    @groupids the goup col idx used for aggregae time stamps
'''
def extracttimes( infiles, outdir, delimeter:str=' ', stbyte=False,
        comments:str='#', nodetype=str, groupids:list=[]):
    return ''

