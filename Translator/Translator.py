#!/usr/bin/env python
# coding: utf-8

# In[8]:


from googletrans import Translator
kata=input('masukkan kata dalam bahasa indo :')
translation=translator.translate(kata, dest='en')
print (kata, '->', translation.text)


# In[9]:


from googletrans import Translator
translator=Translator()
print('TRANSLATOR')

kata=input('Masukkan kata/kalimat: ')
detect=translator.detect(kata)
src=detect.lang
if src == 'idar':
    src == 'id'

lang = {
    'arab' : 'ar',
    'indonesia': 'id',
    'inggris': 'en',
    'italia' : 'it',
    'jepang' : 'ja',
    'jawa' : 'jw',
    'jerman' : 'de',
    'korea': 'ko',
    'malaysia' : 'ms',
    'prancis' : 'fr',
    'spanyol': 'es',
    'sunda' : 'su',
    'thailand' : 'th',
    'turki': 'tr'
    }    
destination=input('Terjemahkan ke bahasa: ') 
dest=lang[destination]
trans=translator.translate(kata, dest, src)
print('')
print(src, '  ->  ', dest)
print(trans.origin, ' -> ', trans.text)


# In[ ]:




