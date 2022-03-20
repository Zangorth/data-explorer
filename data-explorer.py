###########
# Imports #
###########
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pptx import Presentation, util
import matplotlib.ticker as ticker
import streamlit as st
import seaborn as sea
import pandas as pd
import numpy as np
import os

sea.set(style='white', rc={'figure.dpi': 300})

st.set_page_config(page_title='Data Explorer', layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)
sidebar = st.sidebar


#############
# Functions #
#############
def plot_feature(feature, panda, plot=True, fs=(16, 9), missing=False):
    st.write(feature_type)
    if feature_type == 'numeric' and panda[feature].nunique() == bins:
        panda['grouping'] = panda[feature].copy()
        st.write(panda['grouping'].unique())
        
    elif feature_type == 'numeric':
        groups = np.histogram(panda.loc[panda[feature].notnull(), feature], bins=bins)
        panda['grouping'] = pd.cut(panda[feature], bins=groups[1])
        panda['grouping'] = pd.IntervalIndex(panda['grouping']).right
        
    else:
        groups = panda.groupby(feature).size().sort_values(ascending=False).reset_index()[0:bins]
        panda['grouping'] = np.where(panda[feature].isin(groups[feature]), panda[feature], 'other')
    
    st.write(bins)
    st.write(panda['grouping'].nunique())
    fig, hist = plt.subplots(figsize=fs)
    sea.histplot(x='grouping', data=panda, stat='probability', bins=bins, 
                 ax=hist, alpha=1 if target == 'None' else 0.3, color='gray')
    
    plt.xticks(rotation=90)
    plt.xlabel(feature)
    hist.set_ylabel(f'Distribution of {feature} as Percent')
    
    if target != 'None':
        if target_type == 'numeric':
            means = panda.groupby('grouping')[target].agg(['mean', 'sem'])
            dvs = [target]
            means.columns = [f'{target} | mean', f'{target} | sem']
            means = means.reset_index()
        
        else:
            targets = pd.get_dummies(panda[target])
            
            means = targets.groupby(panda['grouping']).agg(['mean', 'sem'])
            dvs = pd.Series(means.columns.get_level_values(0)).unique()
            means.columns = [f'{means.columns.get_level_values(0)[i]} | {means.columns.get_level_values(1)[i]}' for i in range(means.shape[1])]
            means = means.reset_index()
            
        ax = hist.twinx()
        paly = sea.color_palette('Dark2_r', n_colors=len(dvs)).as_hex()
        lines = []
        for i in range(len(dvs)):
            sea.scatterplot(x='grouping', y=f'{dvs[i]} | mean', data=means, color='black', ax=ax)
            ax.errorbar(x='grouping', y=f'{dvs[i]} | mean', yerr=f'{dvs[i]} | sem', data=means, 
                         ls='' if len(dvs) == 1 else '-', label=dvs[i], color=paly[i], alpha=0.6)
            plt.ylabel('')
        
            lines.append(Line2D([0], [0], color=paly[i]))
        
        plt.legend(lines, dvs)
        ax.set_ylabel(f'Average of {target}')
        
    return fig

#############
# Load Data #
#############
if 'panda' not in st.session_state:
    sidebar.header('Data Read')

    query = sidebar.file_uploader('Data File')

    if query is not None:
        st.session_state['panda'] = pd.read_csv(query, encoding='utf-8')
        st.experimental_rerun()

####################
# Data Exploration #
####################
else:
    panda = st.session_state['panda'].copy()

    ###########
    # Options #
    ###########
    # Global Options
    
    with sidebar.expander('Global Options', expanded=True):
        with st.form('global_opts'):
            feature = st.selectbox('Features', panda.columns)
            target = st.selectbox('Target', ['None'] + list(panda.columns))
            partition = st.selectbox('Filter', ['None'] + list(panda.columns))
            outliers = st.checkbox('Outlier Removal')
    
            st.form_submit_button('Submit')
        
    target_type = (None if target == 'None' else 
                   'categorical' if pd.api.types.is_object_dtype(panda[target]) else 'categorical' if panda[target].nunique() == 2 else 
                   'numeric' if pd.api.types.is_numeric_dtype(panda[target]) 
                   else None)
    
    feature_type = ('categorical' if pd.api.types.is_object_dtype(panda[feature]) else 'categorical' if panda[feature].nunique() == 2 else 
                    'numeric' if pd.api.types.is_numeric_dtype(panda[feature]) 
                    else None)

    # Partition Options
    if partition != 'None':
        filter_type = 'categorical' if pd.api.types.is_object_dtype(panda[partition]) else 'numeric'
        
        with sidebar.expander('Filter Options', expanded=True):
            with st.form('filter_opts'):
                if filter_type == 'categorical':
                    filters = st.multiselect('Categories', list(panda[partition].unique()), list(panda[partition].unique()))
    
                else:
                    min_value = st.number_input('Min', min_value=panda[partition].min(), max_value=panda[partition].max(), value=panda[partition].min())
                    max_value = st.number_input('Max', min_value=panda[partition].min(), max_value=panda[partition].max(), value=panda[partition].max())
    
                st.form_submit_button('Submit')

        if filter_type == 'categorical':
            panda = panda.loc[panda[partition].isin(filters)].copy()
        
        else:
            panda = panda.loc[(panda[partition] >= min_value) & (panda[partition] <= max_value)]
              

    # Graph Options
    with sidebar.expander('Graph Options'):
        with st.form('numeric_opts'):
            if feature_type == 'numeric':
                optimal_bins = int(np.ceil(np.log2(len(panda))) + 1)
                bins = st.slider('Number of Bins', 1, min(100, panda[feature].nunique()), min(optimal_bins, panda[feature].nunique()))
            else:
                bins = st.number_input('Top N Categories', min_value=2, max_value=panda[feature].nunique(),
                                       value=panda[feature].nunique() if panda[feature].nunique() <= 12 else 10)
            
            bins = int(bins)
            
            st.form_submit_button('Submit')

    #################
    # Display Plots #
    #################
    st.metric('Percent Missing', panda[feature].isnull().sum()/len(panda))
    
    fig = plot_feature(feature, panda)
    
    st.pyplot(fig)











