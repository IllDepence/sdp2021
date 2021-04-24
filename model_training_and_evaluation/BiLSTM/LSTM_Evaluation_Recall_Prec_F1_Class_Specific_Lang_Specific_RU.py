#!/usr/bin/env python
# coding: utf-8

"""
This file evaluates BiLSTM model instances. 
For this, PDF files and a metadata .csv are read in. The input data is labeled afterwards.
"""

# Import packages

# general imports
import pandas as pd
import os
import csv
import numpy as np
from io import StringIO

# PDFminer imports
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

# Token Vectorization imports
import fasttext.util
import fasttext

# further imports
from numpy import array
import regex as re
import string
import re
from datetime import datetime

# Keras & sklearn imports for ML Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from keras import metrics as km
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, Bidirectional
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D, Embedding
from keras.utils import np_utils
from keras import optimizers
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics

# generate time stamp for stopping runtimes 
time_start = datetime.now()


# load reduced FastText modules
path_to_bins = 'speicher/fasttext_bins/'
print('BEGIN')
# import RU fasttext model in dim. version (100)
ft_ru = fasttext.load_model(path_to_bins+'cc.ru.100.bin')
print('SUCCESS Fasttext Model RU') 
# import UK fasttext model 
ft_uk = fasttext.load_model(path_to_bins+'cc.uk.100.bin')
# import BG fasttext model 
print('SUCCESS Fasttext Model UK') 
ft_bg = fasttext.load_model(path_to_bins+'cc.bg.100.bin')
print('IMPORT of all Fasttext Models DONE')



### Methods 

# Function for removal of newline identifiers
def removePassage(my_str):
    my_str1 = re.sub("\\\\ud", " ", my_str)
    my_str2 = re.sub("\\\\n", " ", my_str1)
    return(my_str2)


# Method for extracting first page of PDF and converting content to String
def extract_page_one(path):
    output_string = StringIO()
    
    with open(path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        interpreter.process_page(list(PDFPage.create_pages(doc))[0])
        return(output_string)


# Method for removing array characters from author lists
def removeAutor(my_str):
    my_str1 = re.sub("\['", "", my_str)
    my_str2 = re.sub("'\]", "", my_str1)
    my_str3 = re.sub("'", "", my_str2)
    return(my_str3)


# Method for extending the input & output vectors by Newlines
def add_newlines(Tokens,Real_Tokens,y_final):
    y_final_REAL = []
    k = 0
    m = 0
    for i in range(len(Tokens)):
        if k == 0:
            j=i
        else:
            if m == 0:
                j = k+1
            else:
                j = m+1
        if Tokens[i] == Real_Tokens[j] : 
            y_final_REAL.append(y_final[i])

            m = j
        else:
            for k in range(j,len(Real_Tokens)): 

                if Real_Tokens[k] == 'NEWLINE':
                    y_final_REAL.append('Sonstiges')

                else:
                    y_final_REAL.append(y_final[i])
                    m=k
                    break

    RealTokens_final = Real_Tokens[:len(y_final_REAL)]
    
    index_title = [i for i, e in enumerate(y_final_REAL) if e == 'I-title']
    end_title = max(index_title)
    
    # label NEWLINES in title as "I-title"        
    for i in range(len(RealTokens_final)):
        if RealTokens_final[i]=='NEWLINE':
            if (y_final_REAL[i+1] =='I-title' or end_title>i) and y_final_REAL[i-1] in ('B-title','I-title'):
                y_final_REAL[i] = 'I-title'

    return(RealTokens_final,y_final_REAL)


# Method for language detection in a String and vectorization using FastText
def detectAndVectorize(tokens_sequence):
    # input is the tokens list of ONE PAPER

    tokens_vectorized = []
    language_handover_var = 0

    try:
        lang = fasttext_detection_model.predict(
            (' '.join(tokens_sequence)).replace('\n', ''))[0]
    except Exception as e:
        print('[FastText] Error at #'+str(i))
        print(e)

    if 'ru' in str(lang):
        language_handover_var = 1
        for i in range(len(tokens_sequence)):
            tokens_vectorized.append(ft_ru.get_word_vector(tokens_sequence[i]))

    elif 'bg' in str(lang):
        for i in range(len(tokens_sequence)):
            tokens_vectorized.append(ft_bg.get_word_vector(tokens_sequence[i]))

    else:  # assume language == uk
        language_handover_var = 2
        for i in range(len(tokens_sequence)):
            tokens_vectorized.append(ft_uk.get_word_vector(tokens_sequence[i]))

    while len(tokens_vectorized) < 1000:
        tokens_vectorized.append(np.zeros(100))

    if len(tokens_vectorized) > 1000:
        del tokens_vectorized[1000:]

    return (np.array(tokens_vectorized),language_handover_var)


# Method for additional feature recognition 
punctuations = '''!()[]{};:'"\<>/?@#$%^&*«»_–~.,-'''

def compute_additional_features(tokens_sequence): 
    # input is the tokens list of ONE PAPER

    tokens = tokens_sequence
    feature_upper = []
    feature_capitalized = []
    feature_autor_format = []
    feature_punctation = []
    feature_newline = []
    feature_array = []
    
    while len(tokens)<1000:
        tokens.append(str(0))
    if len(tokens)>1000:
        del tokens[1000:] 
    for i in range(len(tokens)):
        if tokens[i] !='NEWLINE':
            if str(tokens[i]).isupper():
                feature_upper.append(1)

            else:
                feature_upper.append(0)
        else: 
            feature_upper.append(0)

        if tokens[i] !='NEWLINE':
            if str(tokens[i][0]).isupper():
                feature_capitalized.append(1)

            else:
                feature_capitalized.append(0)
        else: 
            feature_capitalized.append(0)

        if tokens[i] !='NEWLINE':
            if re.match('.\.',str(tokens[i])) != None and str(tokens[i]).isupper():
                feature_autor_format.append(1)

            else:
                feature_autor_format.append(0)
        else: 
            feature_autor_format.append(0)

        if tokens[i] !='NEWLINE':
            if any((c in punctuations) for c in str(tokens[i])):
                feature_punctation.append(1)
            else:
                feature_punctation.append(0)
        else: 
            feature_punctation.append(0)
                
        if tokens[i] =='NEWLINE':
            feature_newline.append(1)
        else: 
            feature_newline.append(0)
    df = pd.DataFrame(list(zip(feature_upper, feature_capitalized,feature_autor_format ,feature_punctation,feature_newline)))  
    feature_array = df.to_numpy(copy=True)
    
    return np.array(feature_array)
    
print('Successful definition of methods') 


# Import Meta Data
root_path='speicher/'
csv_path = root_path+'final_items_15553.csv'
df_meta = pd.read_csv(csv_path, sep = ',')

# part of all papers to analyze i.e. all papers (3 => 1/3 of all papers are analyzed; 1 => whole data set)
part_of_all_to_analyze = 1

path=root_path+'PDFs_15553/'
files = []
file_paths = []
print('Reading directory: '+str(path))     # r=root directory d=directories f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.pdf' in file:
            files.append(file)
            file_paths.append(os.path.join(r, file))
            
print('Path reading DONE')

# Get Core IDs of Files in directory
files_core_id = []
for i in range(len(files)):
    files_core_id.append(int(re.sub('.pdf','',re.sub('Core_ID_','', str(files[i])))))
    

all_pdf_text = [] 
lauf=0


for i in range(int(len(files)/part_of_all_to_analyze)):

    try:
        all_pdf_text.append(extract_page_one(file_paths[i]).getvalue())
        if lauf%100==0:
            print(str((i/int(len(files)/part_of_all_to_analyze))*100)+'%')
    except:
        all_pdf_text.append(' EMPTY ')
    lauf=lauf+1


print('PDF text extraction done - Number of items read in: '+str(len(all_pdf_text)))

# generate time stamp for runtime of PDF read-in 
time_finished_reading = datetime.now()
time_delta_reading = time_finished_reading-time_start
print('Time for init. and PDFs read-in: ',(time_delta_reading.total_seconds() / 60.0), ' min')


# get all titles from meta data with core_ids of fulltext
titles = []
for i in range(int(len(files_core_id)/part_of_all_to_analyze)):
    index = df_meta.index[df_meta['coreId'] == int(files_core_id[i])].tolist()
    if index == []:
        titles.append('Keine Meta Daten gefunden')
    else: 
        index = index[0]
        title_pdf  = df_meta.loc[index,'title']
        titles.append(title_pdf)


# Get autor for the PDFs - (PROBLEM: different format in Meta data and PDF)
autors = []
for i in range(int(len(files_core_id)/part_of_all_to_analyze)):
    index = df_meta.index[df_meta['coreId'] == int(files_core_id[i])].tolist()
    index = index[0]
    autor_pdf  = df_meta.loc[index,'authors']

    autor_pdf = removeAutor(autor_pdf).split(",")
    for j in range(len(autor_pdf)):
        autor_pdf[j] = ' '.join(autor_pdf[j].split()) ## Remove excessive whitespaces
        
    autors.append(autor_pdf)


# Loop for labeling the text tokens
kein_autor = []
kein_titel = []
error_papers = []
ergebnis_tokens = []
ergebnis_label = []
anzahl_papers = lauf

for paper in range(anzahl_papers):
    title = ' '.join(removePassage(titles[paper]).split()).lower() # Remove excces whitespace & convert to lowercase
    title = re.sub("\(","\(",title) 
    title = re.sub("\)","\)",title)

    title_index = re.search(title, ' '.join(all_pdf_text[paper].split()).lower())
    
    if title_index==None:
        kein_titel.append(files_core_id[paper])
    else:
        try:
            Text_pdf_0 = ' '.join(all_pdf_text[paper].split())
            if title_index.start()==0:
                teil_B = ""
            else:
                teil_B = Text_pdf_0[0:title_index.start()-1]
            teil_T = Text_pdf_0[title_index.start():title_index.end()]
            teil_E = Text_pdf_0[title_index.end()+1:len(Text_pdf_0)]

            y_teil1 = np.repeat('Sonstiges',len(teil_B.split()))
            y_teil2 = np.append(['B-title'],np.repeat('I-title',len(teil_T.split())-1))
            y_teil3 = np.repeat('Sonstiges',len(teil_E.split()))

            y_final = np.concatenate((y_teil1, y_teil2 , y_teil3), axis=None)

            ##### Get Text
            all_pdf_text1 = re.sub("\\n"," NEWLINE ",all_pdf_text[paper])
 
            Text_pdf_0_NL = ' '.join(all_pdf_text1.split())

            Tokens = Text_pdf_0.split()
            Labels = y_final
            Real_Tokens = Text_pdf_0_NL.split()

            autors_surname = []
            for i in range(len(autors[paper])):
                if i % 2 == 0:
                    autors_surname.append(autors[paper][i])

            autors_surname_lower = []
            for i in range(len(autors_surname)):
                autors_surname_lower.append(autors_surname[i].lower())
        
            if re.match('.\.',autors[paper][1]) == None:
                autors_forename = []
                for i in range(len(autors[paper])):
                    if i % 2 == 1:
                        autors_forename.append(autors[paper][i].split())

                autors_forename = list(np.concatenate((autors_forename), axis=None))
                autors_forename_lower = []
                for i in range(len(autors_forename)):
                    autors_forename_lower.append(autors_forename[i].lower())

                autors_surname_lower = list(np.concatenate((autors_forename_lower,autors_surname_lower), axis=None))

            Tokens = all_pdf_text[paper].split()
            Tokens_final_lower = []
            for i in range(len(Tokens)):
                Tokens_final_lower.append(Tokens[i].lower())

            vec_autor = []
            for token in Tokens_final_lower:
                line = any(word in token for word in autors_surname_lower)
                vec_autor.append(line)

            index_autor = [i for i, e in enumerate(vec_autor) if e == True]

            if len(index_autor)>(len(autors_surname_lower)):
                diff = len(index_autor) - len(autors_surname_lower)
                dist = []
                for j in range(len(index_autor)):
                    dist.append(abs(index_autor[j]-y_final_REAL.index('B-title')))

                dict1 = dict(zip(dist , index_autor))

                dist.sort(reverse = True)

                for k in range(len(dist[0:diff])):
                    vec_autor[dict1[dist[0:diff][k]]] = False

            for i in range(len(y_final)):
                if vec_autor[i] == True:
                    y_final[i] = 'autor'

            if True not in vec_autor:
                kein_autor.append(files_core_id[paper])
                
            if re.match('.\.',autors[paper][1]) != None:

                index_autor_true = [i for i, e in enumerate(vec_autor) if e == True]

                for w in range(len(index_autor_true)):
                    index = index_autor_true[w]
                    for t in range(index - 4,index + 4):
                        if re.match('.\.',Tokens_final_lower[t]) != None and Tokens[t].isupper():
                            y_final[t] = 'autor'

            RealTokens_final = add_newlines(Tokens,Real_Tokens,y_final)[0]
            y_final_REAL = add_newlines(Tokens,Real_Tokens,y_final)[1]
            ergebnis_label.append(y_final_REAL)
            ergebnis_tokens.append(RealTokens_final)
        except:
            error_papers.append(files_core_id[paper])


print('Length of ergebnis tokens: '+str(len(ergebnis_tokens)))

print('Length of ergebnis labels: '+str(len(ergebnis_label)))


# Start the vectorization of the fulltexts 
print('Start PDF Vectorization')
all_pdfs_vectorized= []

# load language detection module (FastText lid.176.bin)
path_to_lang_model = root_path+'fasttext_bins/lid.176.bin'
fasttext_detection_model = fasttext.load_model(path_to_lang_model)

# Collect indices for languages 
index_list_ru = []
index_list_uk = []
index_list_other = []

for i in range(len(ergebnis_tokens)):
    ft_array, language_indicator = detectAndVectorize(ergebnis_tokens[i])
    all_pdfs_vectorized.append(ft_array)
    # returned var is 0 for BG / else, 1 for RU or 2 for UK
    if language_indicator == 1:
        index_list_ru.append(i)
    if language_indicator == 2:
        index_list_uk.append(i)
    if language_indicator == 0:
        index_list_other.append(i)
    
    if i % 1000 == 0:
        print(str((i/int(len(files)/part_of_all_to_analyze))*100)+'%')

print('End PDF Vectorization')

all_pdfs_vectorized = np.array(all_pdfs_vectorized)

print('Dimensions of vectorized array: '+str(all_pdfs_vectorized.shape))
print('Lengths of index lists by language: RU - '+str(len(index_list_ru))+' | UK - '+str(len(index_list_uk))+' | Other - '+str(len(index_list_other)))
print('Check: Sum of list lengths: '+str(len(index_list_ru)+len(index_list_uk)+len(index_list_other)))


# replace newline vectors with a uniform representation (not 3 diff. vectors for bg, uk and ru)
newline_vec_uk = ft_uk.get_word_vector('NEWLINE')
newline_ru = ft_ru.get_word_vector('NEWLINE')
newline_bg = ft_bg.get_word_vector('NEWLINE')
counter_replaced_ru = 0
counter_replaced_bg = 0

for i in range(len(all_pdfs_vectorized)):
    
    for j in range(len(all_pdfs_vectorized[i])):
        
        compare_array=all_pdfs_vectorized[i][j].copy()
        
        if (compare_array == newline_ru).all():
            all_pdfs_vectorized[i][j] = newline_vec_uk
            counter_replaced_ru +=1
            
        elif (compare_array == newline_bg).all():
            all_pdfs_vectorized[i][j] = newline_vec_uk
            counter_replaced_bg +=1
            
print('RU Replacement Count: '+str(counter_replaced_ru))
print('BG Replacement Count: '+str(counter_replaced_bg))


# add feature vectors to word vectors
all_pdfs_vectorized_features = []

for i in range(len(all_pdfs_vectorized)):
    pdf_vectorized_features = []
    features_in_pdf = compute_additional_features(ergebnis_tokens[i])
    
    for j in range(len(all_pdfs_vectorized[i])):
        pdf_vectorized_features.append(np.append(np.float16(all_pdfs_vectorized[i][j]),np.float16(features_in_pdf[j])))
        
    all_pdfs_vectorized_features.append(np.array(pdf_vectorized_features))

all_pdfs_vectorized_features=np.array((all_pdfs_vectorized_features))
print('Dimensions of array with word vectors and features combined: '+str(all_pdfs_vectorized_features.shape))


# fit labels to the same length as the 1000 character input PDF string
print('Start of fitting labels to 1000 length') 

labels_categorized = []
label_dict = {'Sonstiges':0,'B-title':1,'I-title':2,'autor':3} 

for i in range(len(ergebnis_label)):
    transformed = [label_dict.get(n, n) for n in ergebnis_label[i]]
    
    while len(transformed)<1000:
        transformed.append(np.zeros(1))
    
    if len(transformed)>1000:
        del transformed[1000:] 
    
    categorical = np_utils.to_categorical(transformed)
    labels_categorized.append(np.array(categorical))

labels_categorized = np.array(labels_categorized)

print('Labels shape: '+str(labels_categorized.shape))

indices = range(all_pdfs_vectorized_features.shape[0])

### define 5 splits for KFold split for subsequent cross-validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
run = 1


for train_index, test_index in kf.split(all_pdfs_vectorized_features):

    print('Start of run #'+str(run))
    time_started_training_run = datetime.now()

    # use only items of test set that are of Russian language 
    tests_ru_in_index = []
    for test_item in test_index:
        if test_item in index_list_ru:
            tests_ru_in_index.append(test_item)

    
    x_test = all_pdfs_vectorized_features[tests_ru_in_index]
    y_test = labels_categorized[tests_ru_in_index]

    # LSTM model evaluation 
    print('\n BEGIN LSTM Model Evaluation for run #'+str(run))

    # load model
    model_name = 'Adam-Epochs-30-5Fold-Run-FastText-Detection-'+str(run)+'.h5'
    model_path = 'models/'+ model_name
    model = keras.models.load_model(model_path)

    # make actual prediction
    prediction = model.predict(x_test)

    true_labels_index = np.argmax(y_test, axis=2).flatten()
    # e.g. (3,0) for softmax output [[0,0.2,0,0.8],[0.6,0.1,0.3,0]]
    predicted_labels_index = np.argmax(prediction, axis=2).flatten() 

    class_labels = ['Misc','B-Title','I-Title','Author']

    print(metrics.classification_report(true_labels_index,predicted_labels_index, target_names = class_labels))
    report = metrics.classification_report(true_labels_index,predicted_labels_index, target_names = class_labels, output_dict=True)
    report_name = 'Classification_Report_RU_'+model_name+'_.csv'
    pd.DataFrame(report).to_csv('results/'+report_name)

    run = run +1


time_end = datetime.now()
time_delta_total = time_end - time_start
print('Time total: ',"%.2f" % ((time_delta_total).total_seconds() / 60.0), 'min \n')

print('----- FINISHED -----\n')
