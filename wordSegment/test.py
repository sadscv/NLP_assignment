#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-2 下午9:58
# @Author  : sadscv
# @File    : test.py

from lstm_decoder import seg


seq0 = "《九州缥缈录》是江南的幻想史诗巨著，共6卷。以虚构的“九州”世界为背景，徐徐展开一轴腥风血雨的乱世长卷。"
print(seg(seq0))

seq = False
while not seq:
    print('start loop')
    seq = input('输入句子:').strip()
    result = seg(seq)
    print(result)
    seq = False