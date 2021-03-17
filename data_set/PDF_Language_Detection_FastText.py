
"""
This file detects the languages of given PDF files using fastText and outputs the language occurences.
"""

### Tool: FastText

# general imports 

from io import StringIO
import os
import re
import json
import pandas as pd
import copy
import numpy as np
import fasttext

# PdfMiner imports
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

### functions and PDF read-in

# Method for extracting first page of PDF and converting content to String
def extractPageOne(path):
    output_string = StringIO()
    
    with open(path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        interpreter.process_page(list(PDFPage.create_pages(doc))[0])
        return(output_string)



root_path='data/'

# part of all papers to analyze i.e. all papers (3 => 1/3 of all papers are analyzed; 1 => whole data set)
part_of_all_to_analyze = 1

path=root_path+'PDFs_15553/'
files = []
file_paths = []
print('\n Reading directory: '+str(path))     # r=root directory d=directories f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.pdf' in file:
            files.append(file)
            file_paths.append(os.path.join(r, file))
            
print('Path reading DONE')


# read in PDFs
all_pdf_text = [] 
lauf=0

for i in range(int(len(files)/part_of_all_to_analyze)):

    try:
        all_pdf_text.append(extractPageOne(file_paths[i]).getvalue())
        if lauf%100==0:
            print(str((i/int(len(files)/part_of_all_to_analyze))*100)+'%')
    except:
        all_pdf_text.append(' EMPTY ')
    lauf=lauf+1


print('PDF text extraction done - Number of items read in: '+str(len(all_pdf_text)))

print('\n BEGIN FastText Detection')

# load language detection module (FastText lid.176.bin)
path_to_model=root_path+'fasttext_bins/lid.176.bin'
detection_fasttext_list = []
fasttext_detection_model = fasttext.load_model(path_to_model)

for i in range(len(all_pdf_text)):
    if i%1000==0:
        print('Run #'+str(i))
    try:
        lang = fasttext_detection_model.predict(all_pdf_text[i].replace('\n',''))[0]
        detection_fasttext_list.append(lang[0])
    except KeyboardInterrupt:
        print('Keyboard Interruption triggered')
        sys.exit()
    except Exception as e:
        print('[FastText] Error at #'+str(i))
        print(e)


print('\n Show Results (Language Frequency Counts)')

print('FastText')

fasttext_frequency = {} 

for items in detection_fasttext_list: 
    fasttext_frequency[items] = detection_fasttext_list.count(items) 
    
for key, value in fasttext_frequency.items(): 
    print (str(key) + ' : '+str(value)) 

print('-------- DONE --------')
