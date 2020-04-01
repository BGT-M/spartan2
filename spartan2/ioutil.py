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
  hdfs default delimeter is b'\x01'
'''
def renumberids2(infiles, outdir, delimeter=' ', isbyte=False,
                 comments='#', nodetype=int, colidx=[0, 1]):
    mode = 'b' if isbyte else ''
    users, msgs = {}, {}
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
                line = line.strip()
                if line.startswith(comments):
                    continue
                elems = line.split(delimeter)
                uid = nodetype(elems[colidx[0]])
                bid = nodetype(elems[colidx[1]])
                if uid not in users:
                    users[uid] = len(users)
                if bid not in msgs:
                    msgs[bid] = len(msgs)
                    #t = int(elems[2])
                    #label = int(elems[3])
                nelems = elems.copy()
                nelems[colidx[0]] = str(users[uid]).encode() if isbyte else \
                        str(users[uid])
                nelems[colidx[1]] = str(msgs[bid]).encode() if isbyte else \
                        str(msgs[bid])
                #import ipdb
                #ipdb.set_trace()
                outf.write(delimeter.join(nelems) + b'\n')
            outf.close()
            f.close()

    with open(os.path.join(outdir, 'userid.dict'), 'w'+mode) as f1,\
            open(os.path.join(outdir, 'msgid.dict'), 'w'+mode) as f2:
        for u, val in users.items():
            u = u.decode() if type(u) is bytes else str(u)
            outstr = "{},{}\n".format(u, val)
            x = outstr.encode() if 'b' in mode else outstr
            f1.write(x)
        for m, val in msgs.items():
            m = m.decode() if type(m) is bytes else str(m)
            outstr = "{},{}\n".format(m, val)
            x = outstr.encode() if 'b' in mode else outstr
            f2.write(x)
        f1.close()
        f2.close()
    return len(users), len(msgs)

