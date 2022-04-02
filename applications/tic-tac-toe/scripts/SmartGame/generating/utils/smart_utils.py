#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""smart.py 's utilities
"""

import os
import pickle
import random
import re
import sys
import time

import pandas as pd
import numpy as np

from SmartGame.generating.classdef import TickTackToe

def check_if_about_B(t):
    L = t.length
    for x in range(L):
        for y in range(L):
            if t.board.item(x,y)==-1:
                B = copy(t)
                B.board.itemset((x,y),1)
                if B.checkX(): return B,True
                else: pass
    return t,False
                
def check_if_about(t):
    L = t.length
    for x in range(L):
        for y in range(L):
            if t.board.item(x,y)==-1:
                B = copy(t)
                B.board.itemset((x,y),0)
                if B.checkO(): return B,True
                else: pass
    return t,False

def block(t):
    L = t.length
    for x in range(L):
        for y in range(L):
            if t.board.item(x,y)==-1:
                B = copy(t)
                B.board.itemset((x,y),1)
                if B.checkX(): return ((x,y),True)
                else: pass
    return ((0,0),False)

             
def block_B(t):
    L = t.length
    for x in range(L):
        for y in range(L):
            if t.board.item(x,y)==-1:
                B = copy(t)
                B.board.itemset((x,y),0)
                if B.checkO(): return ((x,y),True)
                else: pass
    return ((0,0),False)
