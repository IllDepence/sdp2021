
#imports

import hashlib 
import copy
import pandas as pd
import os


def getDuplicatesWithIndex(liste):

    dictOfElems = dict()

    for elem in liste:
        # If element exists in dict then increment its value else add it in dict
        if elem in dictOfElems:
            dictOfElems[elem] += 1
        else:
            dictOfElems[elem] = 1    
            
    # Filter key-value pairs in dictionary. Keep pairs whose value is greater than 1 i.e. only duplicate elements from list.
    dictOfElems = { key:value for key, value in dictOfElems.items() if value > 1}
    return dictOfElems



# read subdirectories 

path_folder = '_API_pdf_download'

folders = [] 

# r=root, d=directories, f = files
for r, d, f in os.walk(path_folder):
    for folder in d:
        folders.append(os.path.join(r, folder))


# calculate md5 hash for each file in each subdirectory and append it to the list 'hashes'
hashes = []
all_files = []
for i in folders:
    path = i
    files = []
    file_paths = []
    print('Reading directory: '+str(i))     # r=root directory d=directories f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.pdf' in file:
                files.append(file)
                file_paths.append(os.path.join(r, file))
                
    for f in file_paths:
        hashes.append(hashlib.md5(open(f,'rb').read()).hexdigest())
    for f in files:
        all_files.append(f)


hashes_unique = copy.deepcopy(hashes)
hashes_unique = list(set(hashes_unique)) 

# output number of duplicates according to PDF content
print(len(all_files)-len(hashes_unique)) 

hashes_dict = getDuplicatesWithIndex(hashes)


# create duplicate list with all duplicates except the first unique item
duplicate_list = []

for key,value in hashes_dict.items():
    t=0
    for i, j in enumerate(hashes):
        if j == key:
            if t==0:
                t=t
            else:
                duplicate_list.append(all_files[i])
            t=t+1


# write text file with names of items to be removed from set
with open ('deletion_list.txt', 'w') as file:
    for el in duplicate_list:
        file.write(el+'\n')
    file.close

# remove PDF files
for i in duplicate_list:
    delete_file = i
    file_path = path_folder+delete_file
    os.remove(file_path)

