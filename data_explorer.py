###########
# Imports #
###########
from matplotlib.lines import Line2D as ln
from matplotlib import patches as ptch
from matplotlib import pyplot as plt
import seaborn as sea
import pandas as pd
import numpy as np

################
# Plot Feature #
################
class DataExplorer():
    def __init__(self, figsize=(16, 9), palette='Dark2_r'):
        '''
        Description - 

        Parameters
        ----------
        figsize : tuple
            The dimensions for the returned figure

        Returns
        -------
        None
        '''

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.legend = []
        self.palette = palette

    def hist(self, feature, partition=None, feature_type='auto', outlier=False, show=False):
        '''
        Description - Plots a histogram of the input feature

        Parameters
        ----------
        feature : pd.Series
            The feature to be binned and displayed as a histogram
        feature_type : str
            Whether the input feature is categorical or numerical
            Default is auto, which will attempt to determine this automatically
        outlier : bool
            Indicator for whether outliers should be included / removed from the histogram
            Can only be used with numeric data types
        show : bool
            Indicator for whether the plot should be displayed after running

        Returns
        -------
        A histogram plot of the designated feature
        '''

        # Feature type identification
        feature_type = feature_type if feature_type != 'auto' else self.typing(feature)
        
        # Remove outliers
        if outlier and feature_type == 'numeric':
            feature = feature.loc[(feature >= feature.quantile(0.025)) & (feature <= feature.quantile(0.975))]

        # Identify optimal number of bins
        optimal_bins = int(np.ceil(np.log2(len(feature))) + 1)
        bins = min([optimal_bins, feature.nunique(), 100])

        # Create data for histogram
        if feature_type == 'numeric' and feature.nunique() == bins:
            grouping = feature.copy()

        elif feature_type == 'numeric':
            group = np.histogram(feature.loc[feature.notnull()], bins=bins)
            grouping = pd.cut(feature, bins=group[1])
            grouping = pd.IntervalIndex(grouping).right

        else:
            group = feature.value_counts().sort_values(ascending=False).reset_index()[0:bins]
            grouping = np.where(feature.isin(group[feature.name]), feature, 'other')
            grouping = pd.Series(grouping).sort_values()

        # Plot Histogram
        sea.histplot(x=grouping, stat='count', bins=bins, alpha=0.4, ax=self.ax, color='grey')
        self.ax.set_ylabel(feature.name)
        
        self.ax.set_title(f'Distribution of {feature.name}')

        if show:
            self.fig.show()

        else:
            plt.close()

        # Save data for other functions
        self.feature_type = feature_type
        self.feature = feature.copy()
        self.grouping = grouping.copy()

    def avg(self, kind='mean', show=False):
        '''
        Description - Adds a line representing the average of the distribution to a histogram

        Parameters
        ----------
        kind : str
            The kind of average to be added to the plot. Currently supported averages are mean and median
        show : bool
            Indicator for whether the plot should be displayed after running

        Returns
        -------
        A vertical line indicating the location of the average on the histogram
        '''
        if self.feature_type != 'numeric':
            raise Exception(f'Feature type must be numeric to show the mean\n Current feature is {self.feature.name}')
        
        if kind == 'mean':
            self.ax.axvline(x=self.feature.mean(), color='red', ls='dashed', label='_nolegend_')

        elif kind == 'median':
            self.ax.axvline(x=self.feature.median(), color='red', ls='dashed', label='_nolegend_')

        else:
            raise Exception('The only kinds of averages currently supported are mean and median')
        
        self.legend.append(ln([0], [0], label=kind, color='red', ls='dashed'))

        print(self.legend)

        if show:
            self.fig.show()

        else:
            plt.close()

    def bivariate(self, target, target_type='auto', show=True):
        '''
        Description - Creates a plot showing the relationship between the histogram feature and a target

        Parameters
        ----------
        target : pd.Series
            The dependent variable to plot the feature against
        target_type : str
            Whether the input feature is categorical or numerical
            Default is auto, which will attempt to determine this automatically
        show : bool
            Indicator for whether the plot should be displayed after running

        Returns
        -------
        A vertical line indicating the location of the average on the histogram
        '''
        self.scatter = self.ax.twinx()

        target = target.loc[target.index.isin(self.feature.index)]
        target_type = target_type if target_type != 'auto' else self.typing(target)

        if target_type == 'numeric': 
            means = target.groupby(self.grouping).agg(['mean', 'sem'])
            dvs = [target.name]
            means.columns = [f'{target.name} | mean', f'{target.name} | sem']
            means = means.reset_index()

        else:
            targets = pd.get_dummies(target, drop_first=True if target.nunique()==2 else False)

            means = targets.groupby(self.grouping).agg(['mean', 'sem'])
            dvs = pd.Series(means.columns.get_level_values(0)).unique()
            means.columns = [f'{means.columns.get_level_values(0)[i]} | {means.columns.get_level_values(1)[i]}' for i in range(means.shape[1])]
            means = means.reset_index()

        paly = sea.color_palette(self.palette, n_colors=len(dvs)).as_hex()
        
        lines, lgnd = [], []
        for i in range(len(dvs)):
            sea.scatterplot(x='index', y=f'{dvs[i]} | mean', data=means, 
                            label='_nolegend_', color=paly[i], ax=self.scatter)
            self.scatter.errorbar(x='index', y=f'{dvs[i]} | mean', yerr=f'{dvs[i]} | sem', data=means, 
                                  ls='', color=paly[i], alpha=0.6, label='_nolegend_')
            
            self.legend.append(ln([0], [0], label=dvs[i], marker='.', markersize=10,
                                  markerfacecolor=paly[i], markeredgecolor=paly[i], ls=''))

        if show:
            self.fig.show()

        else:
            plt.close()
        
    def prettify(self, title=None, legend=True, left_label=None, right_label=None, xlabel=None, 
                 xlim=None, left_lim=None, right_lim=None, show=True):
        self.ax.set_title(title if title is not None else '')

        if left_label is not None:
            self.ax.set_ylabel(left_label)

        if right_label is not None:
            self.scatter.set_ylabel(right_label)

        if xlabel is not None:
            self.ax.set_xlabel(xlabel)

        if left_lim is not None:
            self.ax.set_ylim(left_lim)

        if right_lim is not None:
            self.scatter.set_ylim(right_lim)

        if xlim is not None:
            self.ax.set_xlim(xlim)

        if legend:
            self.ax.legend(handles=self.legend, loc='best')

        if show:
            self.fig.show()

        else:
            plt.close()

    def typing(self, x):
        if pd.api.types.is_object_dtype(x) or x.nunique() == 2:
            return 'categorical'
        
        elif pd.api.types.is_numeric_dtype(x):
            return 'numeric'
        
        else:
            raise Exception('Unknown feature type, please input manually')
        
