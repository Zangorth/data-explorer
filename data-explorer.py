###########
# Imports #
###########
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from pptx import Presentation, util
import streamlit as st
import seaborn as sea
import pandas as pd
import numpy as np
import os

sea.set(style='whitegrid', rc={'figure.dpi': 300})

st.set_page_config(page_title='Data Explorer', layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)
sidebar = st.sidebar


#############
# Functions #
#############
def get_numeric(feature, panda, plot=True, fs=(16, 9), missing=False):
    size = np.histogram(panda.loc[panda[feature].notnull(), feature], bins=bins)
    height = size[0]
    original_axis = size[1]
    axis = [(original_axis[i] + original_axis[i+1])/2 for i in range(len(original_axis)-1)]
    size = pd.DataFrame({'cuts': axis, 'size': height})

    if plot:
        fig, hist = plt.subplots(figsize=fs)
        hist.bar(size['cuts'], size['size'], alpha=(0.4 if target != 'None' else 0.8), color='grey',
                  width=((size.cuts.min() + size.cuts.max())/len(size))*0.2)
        hist.set(yticklabels=[], ylabel=None)
        hist.tick_params(left=False)

    if target != 'None':
        panda['grouping'] = np.nan
        groups = 5

        while panda['grouping'].nunique() < 2 and groups <= 50:
            groups += 5
            panda['grouping'] = pd.qcut(panda[feature], groups, duplicates='drop')

        panda['grouping'] = [np.nan if pd.isnull(val) else val.mid for val in panda['grouping']]

        means = panda.groupby('grouping')[target].mean().reset_index()

        se = panda.groupby(['grouping'])[target].sem().reset_index()
        se.columns = ['grouping', 'error']
        se['error'] = 2 * se['error']

        means = means.merge(se, on=['grouping'])

        if plot:
            ax = hist.twinx()
            ax.plot('grouping', target, data=means, color='#148F77')
            ax.errorbar('grouping', target, yerr='error', data=means, ls='', label='', color='black')
            hist.set(yticklabels=[], ylabel=None)
            hist.tick_params(left=False)
            ax.yaxis.tick_left()
            ax.set_ylabel(f'Average of {target}')
            ax.yaxis.set_label_position('left')

            missing = f'\nMissingness: {round((panda[feature].isnull().sum()/len(panda))*100, 1)}%' if missing else ''

            title = f'Distribution of {feature}{missing}' if target == 'None' else f'Relationship between {target} and {feature}{missing}'
            plt.title(title)

            return fig

        else:
            return means

def get_category(feature, panda, plot=True, fs=(16, 9), missing=False):
    missing = f'\nMissingness: {round((panda[feature].isnull().sum()/len(panda))*100, 1)}%' if missing else ''
    size = panda.groupby(feature).size().sort_values(ascending=False).reset_index()[0:bins]
    panda['grouping'] = np.where(panda[feature].isin(size[feature]), panda[feature].astype(str).str.replace(' ', '').str.lower(), 'other')

    size = panda.groupby('grouping').size().reset_index()
    size.columns = ['grouping', 'size']
    size['scaled'] = 0.8*((size['size'] - size['size'].min())/(size['size'].max() - size['size'].min())) + 0.2

    if target == 'None' and plot:
        fig, ax = plt.subplots(figsize=fs)
        ax.bar(size['grouping'], size['size']/size['size'].sum(), color='#148F77')
        ax.set_xticks(np.arange(size['grouping'].nunique()))
        ax.set_xticklabels(size['grouping'], rotation=90)
        plt.title(f'Distribution of {feature}{missing}')
        plt.ylabel('Percent of Accounts')

        return fig

    else:
        means = panda.groupby('grouping')[target].mean().reset_index()
        se = panda.groupby(['grouping'])[target].sem().reset_index()
        se.columns = ['grouping', 'error']
        se['error'] = 2 * se['error']

        means = means.merge(se, on=['grouping'])
        means = means.merge(size, on='grouping')

        if plot:
            fig, ax = plt.subplots(figsize=fs)
            for i in range(len(means)):
                ax.bar(means['grouping'][i], means[target][i], alpha=means['scaled'][i], color='#148F77')

            ax.errorbar('grouping', target, 'error', data=means, ls='', color='black')
            ax.set_xticks(np.arange(size['grouping'].nunique()))
            ax.set_xticklabels(size['grouping'], rotation=90)
            plt.title(f'Relationship between {target} and {feature}{missing}')
            plt.ylabel(f'Average of {target}')

            return fig

        else:
            return means

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
                bins = st.slider('Number of Bins', 1, min(100, panda[feature].nunique()), min(100, panda[feature].nunique()))
            else:
                bins = st.number_input('Top N Categories', min_value=2, max_value=panda[feature].nunique(),
                                       value=panda[feature].nunique() if panda[feature].nunique() <= 12 else 10)
    
            st.form_submit_button('Submit')

    # Powerpoint Options
    with sidebar.expander('PowerPoint Options'):
        with st.form('powerpoint_opts'):
            exclude_features = st.multiselect('Features to Exclude', panda.columns)
            dl = st.form_submit_button('Download PowerPoint')

    #######################
    # PowerPoint Download #
    #######################
    if dl:
        with st.spinner('Generating PowerPoint'):
            prs = Presentation()
            prs.slide_width = util.Inches(16)
            prs.slide_height = util.Inches(9)

            lyt = prs.slide_layouts[0]
            slide = prs.slides.add_slide(lyt)
            title = slide.shapes.title
            subtitle = slide.placeholders[1]

            title.text = 'Insert Title Here'
            subtitle.text = 'Insert Subtitle Here'

        with st.spinner('Sorting Effect Sizes'):
            effect_sizes = pd.DataFrame({'feature': [col for col in panda.columns if col not in exclude_features], 'size': np.nan})
            for feature in effect_sizes['feature']:
                df = panda.copy()

                if pd.api.types.is_object_dtype(df[feature]) or df[feature].nunique() == 2:
                    size = df.groupby(feature).size().sort_values(ascending=False).reset_index()[0:bins]
                    df['grouping'] = np.where(df[feature].isin(size[feature]),
                                              df[feature].astype(str).str.replace(' ', '').str.lower(), 'other')

                    y = df[target].values.ravel()
                    x = pd.get_dummies(df['grouping'])

                    try:
                        model = LinearRegression(fit_intercept=False)
                        model.fit(x, y)

                        effect_sizes.loc[effect_sizes['feature'] == feature, 'size'] = np.abs(max(model.coef_) - min(model.coef_))

                    except ValueError:
                        effect_sizes.loc[effect_sizes['feature'] == feature, 'size'] = -666

                else:
                    if outliers:
                        df = df.loc[(df[feature] >= df[feature].mean() - 2*df[feature].std()) &
                                    (df[feature] <= df[feature].mean() + 2*df[feature].std())]

                    y = df[target].values.ravel()
                    x = df[feature].fillna(df[feature].mean()).values.reshape(-1, 1)

                    scaler = mms()
                    x = scaler.fit_transform(x)

                    try:
                        model = LinearRegression()
                        model.fit(x, y)

                        effect_sizes.loc[effect_sizes['feature'] == feature, 'size'] = np.abs(model.coef_[0])

                    except ValueError:
                        effect_sizes.loc[effect_sizes['feature'] == feature, 'size'] = -666

            effect_sizes = effect_sizes.sort_values('size', ascending=False).reset_index(drop=True)

        i = 0
        for feature in effect_sizes['feature']:
            i += 1
            df = panda.copy()

            with st.spinner(f'Plotting {feature} ({i}/{len(effect_sizes)})'):
                if pd.api.types.is_object_dtype(df[feature]) or df[feature].nunique() == 2:
                    fig = get_category(feature, df, fs=(14, 7.875), missing=True)

                else:
                    if outliers:
                        with st.spinner('Removing Outliers'):
                            df = df.loc[(df[feature] >= df[feature].mean() - 2*df[feature].std()) &
                                        (df[feature] <= df[feature].mean() + 2*df[feature].std())]

                    with st.spinner('Generating Fig'):
                        fig = get_numeric(feature, df, fs=(14, 7.875), missing=True)

                plt.savefig('data-explorer-figure.png', bbox_inches='tight')

                with st.spinner('Adding Fig to PP'):
                    slide = prs.slides.add_slide(prs.slide_layouts[6])
                    img=slide.shapes.add_picture('data-explorer-figure.png', left=util.Inches(2), top=util.Inches(1))
                    os.remove('data-explorer-figure.png')

        with st.spinner('Saving PowerPoint'):
            prs.save('Data-Exploration.pptx')

        st.write(f'Powerpoint Saved At: {os.getcwd()}')

    #################
    # Display Plots #
    #################
    else:
        st.metric('Missingness', panda[feature].isnull().sum()/len(panda))

        if feature_type == 'numeric':
            if outliers:
                panda = panda.loc[(panda[feature] >= panda[feature].mean() - 2*panda[feature].std()) &
                                  (panda[feature] <= panda[feature].mean() + 2*panda[feature].std())]

            fig = get_numeric(feature, panda)

        elif feature_type == 'categorical':
            fig = get_category(feature, panda)

        st.pyplot(fig)











