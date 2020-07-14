#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   basicutil.py
@Desc    :   Basic util functions.
'''

# here put the import lib

def set_trace( isset = True ):
    if isset is True:
        import ipdb; ipdb.set_trace()
    else:
        pass

