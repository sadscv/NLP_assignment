#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-5-1 下午8:45
# @Author  : sadscv
# @File    : test.py
import random

house = [1, 2, 3, 4, 5]
host = [1, 2, 3, 4, 5]
color = {1 : 'red', 2 : 'white',  3 : 'green', 4 : 'yellow',5 : 'blue'}
drink = {1 : 'tea', 2 : 'coffee', 3 : 'milk', 4 : 'beer', 5 : 'water'}
tobacco = {1 : 'PALL MALL', 2 : 'DUNHILL', 3: 'HIBERATE', 4: 'BLUE MASTER', 5 : 'PRINCE'}
pet = {1 : 'dog', 2 : 'bird', 3 : 'cat', 4 : 'horse', 5 : 'fish'}

class Host(object):
    def __init__(self, house, country, drink, tobacoo, pet):
        self.house = house
        self.country = country
        self.drink = drink
        self.tobacoo = tobacoo
        self.pet = pet

class House(object):
    def __init__(self, color, position):
        self.color = color
        self.position = position

def assign():
    list = [1, 2, 3, 4, 5]
    rd = random.randint(1, 5)
    for i in range(5):
        self.



def main():
    assign()