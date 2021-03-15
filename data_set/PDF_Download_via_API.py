# imports

import re
import json
import pandas as pd
import copy
import numpy as np
import urllib
import requests
import matplotlib.pylab as plt
from collections import Counter
from pathlib import Path

# read .json file from CORE collection
data_df_cyrr = pd.read_json('_coredata/core_all_cyr.jsonl', lines=True)

### API Download Section - Functions

# Download function
def download_file(download_url , name, item_range):
    
    # directory check 
    path = '_API_pdf_download/Core_All_Cyr/'+str(item_range)
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path+'/'+name
    
    response = urllib.request.urlopen(download_url)

    file = open(path, 'wb')
    file.write(response.read())
    file.close()
    print("Completed")

def write_error_protocol(error_indx_list, error_coreid_list, item_range):
    
    # directory check 
    path = '_API_pdf_download/Core_All_Cyr/'+str(item_range)
    Path(path).mkdir(parents=True, exist_ok=True)
    path = path+'/_Error_List_'+str(item_range)+'.txt'
    
    error_content = 'The following items of the dataframe could not be downloaded: '+"\n"
    i=0
    for e in error_indx_list:
        error_content = error_content + 'Index: '+str(e) 
        error_content = error_content +' CoreId: '+str(error_coreid_list[i])+"\n" 
        i = i+1
        
    file = open(path, 'wt')
    file.write(error_content)
    file.close()
    print("Error Protocol Completed")


# Loop for download
error_index_list = []
error_paper_list = []

### DOWNLOAD SET OPTIONS (for download in batches) ###
lower_bound = 0
upper_bound = 10000

item_range_path = ''+str(lower_bound)+'_'+str(upper_bound)

# insert your API key here
api_key = ''

for i in range(lower_bound ,upper_bound):

    try:
        core_id = str(data_df_cyrr.iloc[i]['coreId']) 
        url = 'https://core.ac.uk:443/api-v2/articles/get/'+ core_id +'/download/pdf?apiKey='+ api_key
        download_file(url , 'Core_ID_' + core_id + '.pdf', item_range_path)
        
    except (KeyboardInterrupt, SystemExit):
        print('Keyboard Interrupt triggered')
        raise
        
    except:
        print('Error at index # '+str(i))
        error_index_list.append(i)
        error_paper_list.append(data_df_cyrr.iloc[i]['coreId'])

write_error_protocol(error_index_list, error_paper_list, item_range_path)

print('Size of triggered download set: '+str(upper_bound - lower_bound))
print('Error count: '+str(len(error_index_list)))