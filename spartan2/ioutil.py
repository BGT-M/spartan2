import os,sys


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
            if (line.startswith("%") or line.startswith("#")): continue;
            line = line.strip()
            if (" " in line): sep = " "
            if ("," in line): sep = ","
            if (";" in line): sep = ';'
            if ("\t" in line): sep = "\t"
            break
        fp.close()
    return sep

def convert_to_db_type(basic_type):
    if basic_type == int:
        return "INT"
    elif basic_type == str:
        return "TEXT"
    elif basic_type == float:
        return "REAL"
    else:
        return "TEXT"

def loadedgelist(tensor_file, col_ids, col_types):
    '''
    load edge list from file
    format: src det value1, value2......

    return: tuple(col_ids_tuple, col_types_tuple, edge_tuple)
    '''

    if len(col_ids) != len(col_types):
        print("The col_ids' length doesn't match the col_types")
        sys.exit(1)

    sep = get_sep_of_file(tensor_file)
    edgelist = []

    with myreadfile(tensor_file, 'rb') as fin:
        for line in fin:
            line = line.strip()
            if line.startswith("#"):
                continue
            coords = line.split(sep)
            try:
                for i in range(0, len(coords)):
                    coords[i] = col_types[i](coords[i])
            except Exception:
                print("This file content doesn't match the given schema")
                sys.exit(1)
            edgelist.append(tuple(coords))

    for i in range(len(col_types)):
        col_types[i] = convert_to_db_type(col_types[i])

    return tuple(col_ids), tuple(col_types), edgelist
