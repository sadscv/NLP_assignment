#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-5-1 下午8:45
# @Author  : sadscv
# @File    : test.py
import copy
import random as rd
global is_satisfied

host = [1, 2, 3, 4, 5]
house_position = {1 : 1, 2 : 2, 3 : 3, 4 : 4, 5 : 5}
house_color = {1 : 'red', 2 : 'white',  3 : 'green', 4 : 'yellow',5 : 'blue'}
drink = {1 : 'tea', 2 : 'coffee', 3 : 'milk', 4 : 'beer', 5 : 'water'}
tobacco = {1 : 'PALL MALL', 2 : 'DUNHILL', 3: 'HIBERATE', 4: 'BLUE MASTER', 5 : 'PRINCE'}
pet = {1 : 'dog', 2 : 'bird', 3 : 'cat', 4 : 'horse', 5 : 'fish'}
country = {1 : 'England', 2 : 'Sweden', 3 : 'Denmark', 4 : 'Norway', 5 : 'German'}

class Host(object):
    def __init__(self, house_position, house_color, country, drink, tobacco, pet):
        self.house_position = house_position
        self.house_color = house_color
        self.country = country
        self.drink = drink
        self.tobacco = tobacco
        self.pet = pet

def print_p(person):
    print(', '.join(['%s:%s' % item for item in person.__dict__.items()]))

def assign(_house_position, _house_color, _country, _drink, _tobacco, _pet):
    list = [_house_position, _house_color, _country, _drink, _tobacco, _pet]
    def get_one(list, type):
        while(len(list[type]) > 0):
            random = rd.randint(1, 5)
            if random not in list[type]:
                continue
            else:
                pop = list[type].pop(random)
                break
        return pop

    p1 = Host(get_one(list, 0), get_one(list, 1), get_one(list, 2), get_one(list, 3), get_one(list, 4), get_one(list, 5))
    p2 = Host(get_one(list, 0), get_one(list, 1), get_one(list, 2), get_one(list, 3), get_one(list, 4), get_one(list, 5))
    p3 = Host(get_one(list, 0), get_one(list, 1), get_one(list, 2), get_one(list, 3), get_one(list, 4), get_one(list, 5))
    p4 = Host(get_one(list, 0), get_one(list, 1), get_one(list, 2), get_one(list, 3), get_one(list, 4), get_one(list, 5))
    p5 = Host(get_one(list, 0), get_one(list, 1), get_one(list, 2), get_one(list, 3), get_one(list, 4), get_one(list, 5))
    # print_p(p1)
    # print_p(p2)
    # print_p(p3)
    # print_p(p4)
    # print_p(p5)
    return [p1, p2, p3, p4 ,p5]


def constrained(person, conditions):
    for p in person:
        for condition in conditions:
            if getattr(p, condition[0][0]) == condition[0][1]:
                if getattr(p, condition[1][0]) != condition[1][1]:
                    is_satisfied = 0
                    return 0
    print('#' * 80)
    return 1



def main():
    is_satisfied = -1
    data = (house_position, house_color, country, drink, tobacco, pet)
    while(is_satisfied != 1):
        _data = copy.deepcopy(data)
        person = assign(_data[0], _data[1], _data[2], _data[3], _data[4], _data[5])
        condition_1 = [('country', 'England'), ('house_color', 'red' )]
        condition_2 = [('country', 'Sweden'), ('pet', 'dog')]
        condition_3 = [('country', 'Denmark'), ('drink', 'tea')]
        condition_4 = [('house_color', 'green'), ('drink', 'coffee')]
        condition_5 = [('tobacco', 'PALL MALL'), ('pet', 'bird')]
        condition_6 = [('house_position', '3'), ('drink', 'milk')]
        condition_7 = [('house_color', 'yellow'), ('tobacco', 'DUNHILL')]
        condition_8 = [('country', 'Norway'), ('house_position', '1')]
        condition_9 = [('tobacco', 'BLUE MASTER'), ('drink', 'beer')]
        condition_10 = [('country', 'German'), ('tobacco', 'PRINCE')]
        conditions = [condition_1, condition_2, condition_3, condition_4, condition_5, condition_6, condition_7,condition_8, condition_9, condition_10]
        # conditions.append()
        is_satisfied = constrained(person, conditions)
        print(is_satisfied)
    else:
        print('@' * 100)


if __name__ == '__main__':
    main()
