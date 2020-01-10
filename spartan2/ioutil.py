import os
import sys


def myreadfile(fnm, mode):
    if 'w' in mode:
        if '.gz' == fnm[-3:]:
            import gzip
            f = gzip.open(fnm, mode)
        else:
            f = open(fnm, mode)
    elif 'r' in mode:
        if '.gz' == fnm[-3:]:
            fnm = fnm[:-3]
        if os.path.isfile(fnm):
            f = open(fnm, mode)
        elif os.path.isfile(fnm+'.gz'):
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


def renumberids(fnm, ofnm, path, respath, delimeter, nodetype=int):
    users, msgs = {}, {}
    with myreadfile(fnm, 'rb') as f, myreadfile(respath+ofnm, 'wb') as outf:
        for line in f:
            line = line.strip()
            elems = line.split(delimeter)
            uid = nodetype(elems[0])
            bid = nodetype(elems[1])
            if uid not in users:
                users[uid] = len(users)
            if bid not in msgs:
                msgs[bid] = len(msgs)
            #t = int(elems[2])
            #label = int(elems[3])
            pos = line.find(' ')
            pos = line.find(' ', pos+1)
            outf.write('{} {} {}\n'.format(
                users[uid], msgs[bid], line[pos+1:]))
        outf.close()
        f.close()
    with open(respath+'userid.dict', 'wb') as f1,\
            open(respath+'msgid.dict', 'wb') as f2:
        for u, val in users.items():
            f1.write("{},{}\n".format(u, val))
        for m, val in msgs.items():
            f2.write("{},{}\n".format(m, val))
        f1.close()
        f2.close()
    return len(users), len(msgs)
