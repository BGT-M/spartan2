import os
import sys


def myreadfile(fnm, mode):
    if '.gz' == fnm[-3:]:
        fnm = fnm[:-3]
    if os.path.isfile(fnm):
        f = open(fnm, 'r')
    elif os.path.isfile(fnm+'.gz'):
        import gzip
        f = gzip.open(fnm+'.gz', 'rb')
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
