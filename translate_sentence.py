#!/usr/bin/env python
# coding: utf-8

#pip install googletrans

from googletrans import Translator
def translate_sentence(english):
    tr_result=Translator().translate(str(english), dest='ko').text
    return tr_result

# 예시: translate('hello') => 여보세요

