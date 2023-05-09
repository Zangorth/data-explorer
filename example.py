import sklearn.datasets as dts
from scipy import stats
import pandas as pd
import numpy as np

iris, y = dts.load_iris(as_frame=True, return_X_y=True)
iris['setosa'] = np.where(y == 0, 'setosa', 'not setosa')
iris['target'] = y.map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

iris['petal_length_oridnal'] = np.where(iris['petal length (cm)'] < 3, '1: short', '2: typical')
iris['petal_length_oridnal'] = np.where(iris['petal length (cm)'] > 5, '3: long', iris['petal_length_oridnal'])

diabetes, y = dts.load_diabetes(as_frame=True, return_X_y=True)
diabetes['target'] = y.copy()

from data_explorer import DataExplorer

de = DataExplorer()
de.hist(iris['petal_length_oridnal'])
de.bivariate(iris['setosa'], show=False)
de.prettify(right_label='Percent', left_label='Count', title='Probability of Setosa by Petal Length')

de = DataExplorer()
de.hist(iris['petal_length_oridnal'])
de.bivariate(iris['target'])


de = DataExplorer()
de.hist(diabetes['age'])
de.avg(show=True)
de.bivariate(diabetes['target'])
de.prettify(right_lim=(0, 300), right_label='Diabetes', left_label='Age', xlabel='Age (Normalized)')