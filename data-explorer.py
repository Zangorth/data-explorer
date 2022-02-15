from matplotlib import pyplot as plt
from pptx import Presentation, util
import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title='Data Explorer', layout='wide')

sidebar = st.sidebar

if 'panda' not in st.session_state:
    sidebar.header('Data Read')

    query = sidebar.file_uploader('Data File')

    if query is not None:
        st.session_state['panda'] = pd.read_csv(query, encoding='utf-8')
        st.experimental_rerun()

else:
    panda = st.session_state['panda'].copy()

    sidebar.subheader('Gloptal Options')

    with sidebar.form('global_opts'):
        feature = st.selectbox('Features', panda.columns)
        target = st.selectbox('Target', ['None'] + list(panda.columns))

        partition = st.selectbox('Filter', ['None'] + list(panda.columns))

        st.form_submit_button('Submit')

    feature_type = 'categorical' if pd.api.types.is_object_dtype(panda[feature]) else 'categorical' if panda[feature].nunique() == 2 else 'numeric' if pd.api.types.is_numeric_dtype(panda[feature]) else None

    if partition != 'None':
        filter_type = 'categorical' if pd.api.types.is_object_dtype(panda[partition]) else 'categorical' if panda[partition].nunique() == 2 else 'numeric'

        with sidebar.form('filter_opts'):
            if filter_type == 'categorical':
                filters = st.multiselect('Categories', list(panda[partition].dropna().unique()), list(panda[partition].dropna().unique()))

            else:
                left, right = st.columns(2)
                equality = left.selectbox('>=<', ['>', '<', '='])
                value = right.number_input('value')

            st.form_submit_button('Submit')

        if filter_type == 'categorical' and filters != ['All']:
            panda = panda.loc[panda[partition].astype(str).isin(filters)].copy()

    st.metric('Missingness', panda[feature].isnull().sum()/len(panda))

    if feature_type == 'numeric':
        sidebar.subheader('Graph Options')
        with sidebar.form('numeric_opts'):
            bins = st.slider('Number of Bins', 1, min(100, panda[feature].nunique()), min(panda[feature].nunique(), 20))

            alpha = st.slider('Histogram Opacity', 0, 100, value=20 if target != 'None' else 80)
            alpha = alpha/100

            width = st.slider('Histogram Width', 0, 100, value=20)
            width = width/100

            st.form_submit_button('Submit')

        size = np.histogram(panda.loc[panda[feature].notnull(), feature], bins=bins)
        height = size[0]
        original_axis = size[1]
        axis = [(original_axis[i] + original_axis[i+1])/2 for i in range(len(original_axis)-1)]
        size = pd.DataFrame({'cuts': axis, 'size': height})


        fig, hist = plt.subplots(figsize=(16, 9))
        hist.bar(size['cuts'], size['size'], alpha=alpha, color='grey',
                  width=((size.cuts.min() + size.cuts.max())/len(size))*width)
        hist.set(yticklabels=[], ylabel=None)
        hist.tick_params(left=False)

        if target != 'None':
            panda['grouping'] = np.nan
            for i in range(len(original_axis)-1):
                panda.loc[panda[feature] >= original_axis[i], 'grouping'] = axis[i]

            means = panda.groupby('grouping')[target].mean().reset_index()

            se = panda.groupby(['grouping'])[target].sem().reset_index()
            se.columns = ['grouping', 'error']
            se['error'] = 2 * se['error']

            means = means.merge(se, on=['grouping'])

            ax = hist.twinx()
            ax.plot('grouping', target, data=means, color='#148F77')
            ax.errorbar('grouping', target, yerr='error', data=means, ls='', label='', color='black')
            hist.set(yticklabels=[], ylabel=None)
            hist.tick_params(left=False)
            ax.yaxis.tick_left()
            ax.set_ylabel(f'Average of {target}')
            ax.yaxis.set_label_position('left')

        title = f'Distribution of {feature}' if target == 'None' else f'Relationship between {target} and {feature}'
        plt.title(title)

        st.pyplot(fig)

    elif feature_type == 'categorical':
        sidebar.subheader('Graph Options')

        with sidebar.form('categorical_opts'):
            bins = st.number_input('Top N Categories', min_value=2, max_value=panda[feature].nunique(),
                                   value=panda[feature].nunique() if panda[feature].nunique() <= 12 else 10)

            st.form_submit_button('Submit')

        size = panda.groupby(feature).size().sort_values(ascending=False).reset_index()[0:bins]
        panda['grouping'] = np.where(panda[feature].isin(size[feature]), panda[feature].astype(str).str.replace(' ', '').str.lower(), 'other')

        size = panda.groupby('grouping').size().reset_index()
        size.columns = ['grouping', 'size']
        size['scaled'] = 0.8*((size['size'] - size['size'].min())/(size['size'].max() - size['size'].min())) + 0.2

        if target == 'None':
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.bar(size['grouping'], size['size']/size['size'].sum(), color='#148F77')
            ax.set_xticks(np.arange(size['grouping'].nunique()))
            ax.set_xticklabels(size['grouping'], rotation=90)
            plt.title(f'Distribution of {feature}')
            plt.ylabel('Percent of Accounts')

        else:
            means = panda.groupby('grouping')[target].mean().reset_index()
            se = panda.groupby(['grouping'])[target].sem().reset_index()
            se.columns = ['grouping', 'error']
            se['error'] = 2 * se['error']

            means = means.merge(se, on=['grouping'])
            means = means.merge(size, on='grouping')

            fig, ax = plt.subplots(figsize=(16, 9))
            for i in range(len(means)):
                ax.bar(means['grouping'][i], means[target][i], alpha=means['scaled'][i], color='#148F77')

            ax.errorbar('grouping', target, 'error', data=means, ls='', color='black')
            ax.set_xticks(np.arange(size['grouping'].nunique()))
            ax.set_xticklabels(size['grouping'], rotation=90)
            plt.title(f'Relationship between {target} and {feature}')
            plt.ylabel(f'Average of {target}')

        st.pyplot(fig)

    ppflag = False
    if sidebar.button('Download as Powerpoint'):
        prs = Presentation()
        prs.slide_width = util.Inches(16)
        prs.slide_height = util.Inches(9)

        lyt = prs.slide_layouts[0]
        slide = prs.slides.add_slide(lyt)
        title = slide.shapes.title
        subtitle = slide.placeholders[1]

        title.text = 'Insert Title Here'
        subtitle.text = 'Insert Subtitle Here'

        prs.save('Data-Exploration.pptx')

        ppflag = True

    if ppflag:
        st.write(os.getcwd())












