###########
# Imports #
###########
from pydataset import data
import pandas as pd
import os

os.chdir(r'C:\Users\Samuel\Google Drive\Portfolio\Data Explorer')

#############
# Data Pull #
#############

for dataset in ['Titanic', 'Salaries', 'Ketchup', 'politicalInformation']:
    panda = data(dataset)
    
    panda.to_csv(f'Data\\{dataset}.csv', index=False, encoding='utf8')

