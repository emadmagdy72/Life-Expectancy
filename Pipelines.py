import pandas as pd
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from colorama import Fore
from sklearn.ensemble import (RandomForestRegressor ,
                                HistGradientBoostingRegressor,
                                ExtraTreesRegressor)

from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pycountry_convert as pc
from sklearn.model_selection import cross_val_score, cross_validate


class ConvertCountry_Continent(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self
    

    def transform(self, X_, y=None):

        def country_to_continent(country_name):
            country_alpha2 = pc.country_name_to_country_alpha2(country_name)
            try:
                country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)

            except Exception as e:
                # Special Case: Timor Leste
                if e == "Invalid Country Alpha-2 code: \'TL\'":
                    country_continent_code = 'AS'
                else:
                    country_continent_code = 'EU'
            country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
            return country_continent_name
    
        X = X_.copy()

        X.loc[:,self.column] = X[self.column].str.split('(',expand=True)[0].str.strip()
        correct_name_country = {'Korea':'North Korea',
                        'Macedonia':'North Macedonia',
                        'TL':'East Timor'}
    
        for old, new in correct_name_country.items():
            idx = X[X[self.column].str.contains(old)].index
            X.loc[idx,self.column] = new
    
    
        X[self.column] = X[self.column].apply(country_to_continent)    

        return X

cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer = SimpleImputer(strategy="median")

cat_pipeline_ord = Pipeline([
        ('cat_imputer', cat_imputer),
        ('ord_enc',OrdinalEncoder())
])
cat_pipeline_ohe = Pipeline([
    ("cat_imputer", cat_imputer),
    ('ohe_enc', OneHotEncoder())
])

num_pipeline_std = Pipeline([
    ("num_imputer", num_imputer),
    ('standard_scale', StandardScaler())
])

num_pipeline_minmax = Pipeline([
    ("num_imputer", num_imputer),
    ('minmax_scale', MinMaxScaler())
])
convert_continent_pipeline = Pipeline([
    'convert', ConvertCountry_Continent('Country')
    ])


pca = PCA(n_components=12)

def Full_pipeline1 (X_train):
    '''
    1. Convert country to its continent
    2. Impute numeric_cols with median and then std scaler
    3. Ordinal_encoder cat_cols
    4. Drop: `GDP`, `thinness 5-9 years`, `under-five deaths`, 'Year', 'Country'
    '''

    

    cat_cols = ['Status','Country']
    numeric_cols = list(X_train.select_dtypes(exclude='object').columns)
    drop_cols = ['GDP', 'thinness 5-9 years', 'under-five deaths', 'Year']


    full_pipeline = ColumnTransformer([
        ('convert',convert_continent_pipeline,'Country'),
        ('cat',cat_pipeline_ord,cat_cols),
        ('numeric', num_pipeline_std,numeric_cols),
        ('drop_cols',"drop",drop_cols)
    ], remainder='passthrogh')

    fill_na = SimpleImputer(strategy='median')

    X_train = full_pipeline.fit_transform(X_train)
    
    return  X_train , full_pipeline 

def Full_pipeline2 (X_train):
    '''
    1. Impute numeric_cols with median and then std scaler
    2. Ordinal_encoder cat_cols
    3. Drop: `GDP`, `thinness 5-9 years`, `under-five deaths`, 'Year', 'Country', 'Schooling'
    '''

    

    cat_cols = ['Status']
    numeric_cols = list(X_train.select_dtypes(exclude='object').columns)
    for col in ['GDP', 'thinness 5-9 years', 'under-five deaths', 'Year','Schooling']:
        numeric_cols.remove(col)


    full_pipeline = ColumnTransformer([
        
        ('cat',cat_pipeline_ord,cat_cols),
        ('numeric', num_pipeline_std,numeric_cols),
        ('drop_cols', "drop", ['GDP', 'thinness 5-9 years', 'under-five deaths', 'Year','Country','Schooling'])
    ], remainder='passthrough')


    X_train = full_pipeline.fit_transform(X_train)
    
    return  X_train , full_pipeline 




def Full_pipeline3 (X_train):
    '''
    1. Impute numeric_cols with median and then std scaler
    2. one hot encoding cat_cols
    3. Drop: `GDP`, `thinness 5-9 years`, `under-five deaths`, 'Year', 'Country'
    4. Drop Schooling 
    '''

    

    cat_cols = ['Status']
    numeric_cols = list(X_train.select_dtypes(exclude='object').columns)
    for col in ['GDP', 'thinness 5-9 years', 'under-five deaths', 'Year','Schooling']:
        numeric_cols.remove(col)


    full_pipeline = ColumnTransformer([
        
        ('cat',cat_pipeline_ohe,cat_cols),
        ('numeric', num_pipeline_std,numeric_cols),
        ('drop_cols', "drop", ['GDP', 'thinness 5-9 years', 'under-five deaths', 'Year','Country','Schooling'])
    ], remainder='passthrough')


    X_train = full_pipeline.fit_transform(X_train)
    
    return  X_train , full_pipeline 

def Full_pipeline4 (X_train):
    '''
    1. Impute numeric_cols with median and then MinMax scaler
    2. one hot encoding cat_cols
    3. Drop: `GDP`, `thinness 5-9 years`, `under-five deaths`, 'Year', 'Country'
    4. Drop Schooling 
    '''

    

    cat_cols = ['Status']
    numeric_cols = list(X_train.select_dtypes(exclude='object').columns)
    for col in ['GDP', 'thinness 5-9 years', 'under-five deaths', 'Year','Schooling']:
        numeric_cols.remove(col)


    full_pipeline = ColumnTransformer([
        
        ('cat',cat_pipeline_ohe,cat_cols),
        ('numeric', num_pipeline_minmax,numeric_cols),
        ('drop_cols', "drop", ['GDP', 'thinness 5-9 years', 'under-five deaths', 'Year','Country','Schooling'])
    ], remainder='passthrough')


    X_train = full_pipeline.fit_transform(X_train)
    
    return  X_train , full_pipeline 


def Full_pipeline5 (X_train):
    '''
    1. Impute numeric_cols with median and then std scaler
    2. Ordinal_encoder cat_cols
    3. Drop: `GDP`, `thinness 5-9 years`, `under-five deaths`, 'Year', 'Country', 'Schooling'
    4. PCA : reduce feature 12 columns 
    '''

    
    
    cat_cols = ['Status']
    numeric_cols = list(X_train.select_dtypes(exclude='object').columns)
    for col in ['GDP', 'thinness 5-9 years', 'under-five deaths', 'Year','Schooling']:
        numeric_cols.remove(col)

    pca_cols = numeric_cols.append('Status')
    
    full_pipeline = ColumnTransformer([
        
        ('cat',cat_pipeline_ord,cat_cols),
        ('numeric', num_pipeline_std,numeric_cols),
        ('drop_cols', "drop", ['GDP', 'thinness 5-9 years', 'under-five deaths', 'Year','Country','Schooling']),
        ('pca',pca,pca_cols)
    ], remainder='passthrough')


    X_train = full_pipeline.fit_transform(X_train)
    
    return  X_train , full_pipeline 



def Production_pipeline(df,full_pipeline):
    """
    - df : dataframe which will come from the user
    """
    X_test = df.drop('Life expectancy',axis=1)
    
    
    X_test = full_pipeline.transform(X_test)
    
    
    return X_test



def compute_pipeline(x,y,models,model_name,n,cv=5,scoring = ['r2','neg_mean_absolute_error']):
    """
    x : x data
    y : y data
    n : pipeline number
    cv : number of cross validations (defult = 5)
    scoring : list of required metrics
    """
    c = 0
    pipeline_dict = {}

    print('Pipeline {} : '.format(n))
    print('------------\n\n')

    for model in models:

        start = time.time()
        scores = cross_validate(model, x, y, cv=cv, scoring=scoring)
        end = time.time()
        
        total_time = end - start

        print(f'Time of {model_name[c]} on {cv} cross validaions : {total_time}')

        print(f'Mean r2 score of {model_name[c]} on {cv} cross validaions : ', scores['test_r2'].mean())

        print(f'Mean neg_mean_absolute_error score of {model_name[c]} on {cv} cross validaions : ', 
              -scores['test_neg_mean_absolute_error'].mean())

        c += 1
        print('----------------------------------------------------------------------------------------------\n')
        



