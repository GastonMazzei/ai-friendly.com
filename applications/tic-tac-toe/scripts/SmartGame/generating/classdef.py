#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""a class that is a tick-tack-toe board
   instantiated with the parameter "L", 
   i.e. the length of the square-grid
"""

import os
import pickle
import random
import re
import sys
import time

import pandas as pd
import numpy as np

class TickTackToe:
    def __init__(self, L=3, pL=3):
         
        if pL>L:
            raise ValueError("ERROR 0: Pattern length can't exceed grid length!")
        else:
            self.length = L
            self.patternLength = pL
        self.board = np.matrix([[-1]*L for x in range(L) ])

    def movesX(self):
        L = self.length
        if bool(np.isin(-1,self.board)):
            temp = []
            temp2 = self.board.ravel().tolist()[0]
            for x in range(L**2):
                if temp2[x]==-1:
                    temp.append((x//L,x%L))    
            (j,k) = random.choice(temp)     
            self.board.itemset((j,k),1)
            return
        else: raise Exception('ERROR 1: No more moves allowed!') 

    def movesO(self):
        L = self.length
        if bool(np.isin(-1,self.board)):
            temp = []
            temp2 = self.board.ravel().tolist()[0]
            for x in range(L**2):
                if temp2[x]==-1:
                    temp.append((x//L,x%L))    
            (j,k) = random.choice(temp)     
            self.board.itemset((j,k),0)
            return
        else: raise Exception('ERROR 1: No more moves allowed!') 

    def movesO_to(self,j,k):
        if self.board.item(j,k)==-1:
            self.board.itemset((j,k),0)
            return
        else:
            if bool(np.isin(-1,self.board)):
                return 'ERR1: Invalid Move!'
            else: raise Exception('ERROR 1: No more moves allowed!') 


    def movesX_to(self,j,k):
        if self.board.item(j,k)==-1:
            self.board.itemset((j,k),1)
            return
        else:
            if bool(np.isin(-1,self.board)):
                return 'ERR1: Invalid Move!'
            else: raise Exception('ERROR 1: No more moves allowed!') 

    def checkO(self):
        pL = self.patternLength
        L = self.length
        for x in range(L):
            for y in range(L):
                if self.board.item(x,y)==0:
                    tempHR = 1
                    tempHL = 1
                    tempVU = 1
                    tempVD = 1
                    tempDUL = 1 
                    tempDDL = 1 
                    tempDDR = 1 
                    tempDUR = 1
                    for q in range(pL):
                        try:
                            #horizontal right                            
                            if self.board.item(x,y+q)==0:                 
                                tempHR += 1
                        except:
                            pass
                        try:
                            #horizontal left                            
                            if self.board.item(x,y-q)==0 and y-q>=0:                 
                                tempHL += 1
                        except:
                            pass
                        try:
                            #vertical up                            
                            if self.board.item(x-q,y)==0 and x-q>=0:                 
                                tempVU += 1
                        except:
                            pass
                        try:
                            #vertical down                            
                            if self.board.item(x+q,y)==0:                 
                                tempVD += 1
                        except:
                            pass
                        try:
                            #diagonal (towards) downleft                            
                            if self.board.item(x+q,y-q)==0 and y-q>=0:                 
                                tempDDL += 1
                        except:
                            pass
                        try:
                            #diagonal (towards) downrigh                            
                            if self.board.item(x+q,y+q)==0:                 
                                tempDDR += 1
                        except:
                            pass
                        try:
                            #diagonal (towards) upleft                            
                            if self.board.item(x-q,y-q)==0 and y-q>=0 and x-q>=0:                 
                                tempDUL += 1
                        except:
                            pass
                        try:
                            #diagonal (towards) upright                            
                            if self.board.item(x-q,y+q)==0 and x-q>=0:                 
                                tempDUR += 1
                        except:
                            pass
                    resultados = [tempHR, tempHL, tempVU, tempVD, tempDDL, tempDDR, tempDUL, tempDUR]
                    for i in range(len(resultados)):
                        if resultados[i]==pL+1:
                            return True
        return False

    def checkX(self):
        pL = self.patternLength
        L = self.length
        for x in range(L):
            for y in range(L):
                if self.board.item(x,y)==1:
                    tempHR = 1
                    tempHL = 1
                    tempVU = 1
                    tempVD = 1
                    tempDUL = 1 
                    tempDDL = 1 
                    tempDDR = 1 
                    tempDUR = 1
                    for q in range(pL):
                        try:
                            #horizontal right                            
                            if self.board.item(x,y+q)==1 :                 
                                tempHR += 1
                        except:
                            pass
                        try:
                            #horizontal left                            
                            if self.board.item(x,y-q)==1 and y-q>=0:                 
                                tempHL += 1
                        except:
                            pass
                        try:
                            #vertical up                            
                            if self.board.item(x-q,y)==1 and x-q>=0:                 
                                tempVU += 1
                        except:
                            pass
                        try:
                            #vertical down                            
                            if self.board.item(x+q,y)==1 :                 
                                tempVD += 1
                        except:
                            pass
                        try:
                            #diagonal (towards) downleft                            
                            if self.board.item(x+q,y-q)==1 and y-q>=0:                 
                                tempDDL += 1
                        except:
                            pass
                        try:
                            #diagonal (towards) downrigh                            
                            if self.board.item(x+q,y+q)==1 :                 
                                tempDDR += 1
                        except:
                            pass
                        try:
                            #diagonal (towards) upleft                            
                            if self.board.item(x-q,y-q)==1 and y-q>=0 and x-q>=0:                 
                                tempDUL += 1
                        except:
                            pass
                        try:
                            #diagonal (towards) upright                            
                            if self.board.item(x-q,y+q)==1 and x-q>=0:                 
                                tempDUR += 1
                        except:
                            pass
                    resultados = [tempHR, tempHL, tempVU, tempVD, tempDDL, tempDDR, tempDUL, tempDUR]
                    for i in range(len(resultados)):
                        if resultados[i]==pL+1:            
                            return True
        return False                       
