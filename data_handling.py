import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


def load_data(path):
    
    return pd.read_csv(path)

data = load_data("Melbourne_housing_FULL.csv")


data.drop(["Lattitude", "Longtitude",'Address', 'Method', 'SellerG', 'Date', 'Postcode', 'Regionname', 
           'Propertycount'],axis="columns", inplace=True)

def data_separation(drop = "no"):
    #removiving target column from training data
    new_data = data.copy()
    #if yes all NAN values will be dropped before data split to x and y
    if drop == "yes":
        new_data.dropna(axis=0, inplace=True)
        y = new_data["Price"]
        y.dropna(axis=0, inplace=True)
        
    
    y = new_data["Price"]
    x = new_data.drop(["Price"], axis='columns')
   
    return x,y


#a function that get list of missing values categorical and NAN values
def missing_values_list(x):
    #getting list of missing data
    missing_values = [col for col in x.columns
                    if x[col].isnull().any()]

    #getting list of categorical data
    missing_dtypes = [cols for cols in x.columns
            if x[cols].dtype == "object"]
    return missing_values , missing_dtypes

#function to input missing values using simpleimputer
def imputer(x_value,value = "x"):
    #setting up imputer
    imputer = SimpleImputer(strategy='most_frequent')

    #inputation
    imputed_x = x_value.copy()
    if value == "y":
        imputed_x = imputer.fit_transform(imputed_x)
    else:
        imputed_x = pd.DataFrame(imputer.fit_transform(imputed_x))

    if value == "y":
        pass
    else:
        # Imputation removed column names; put them back
        imputed_x.columns = x_value.columns
    
    return imputed_x

def encoder(x_value):
    #setting up encoder
    ordinal_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # Make copy to avoid changing original data preprocess data
    label_x = imputer(x_value)
    __ ,missing_dtypes = missing_values_list(x_value)
    
    # Apply ordinal encoder to each column with categorical data
    label_x[missing_dtypes] = ordinal_enc.fit_transform(label_x[missing_dtypes])
    
    return label_x




# Separate inputs and target
def y_missing():
    #reseting index to avoid nan values to be created
    x ,y = data_separation()
    label_x = encoder(x).reset_index(drop=True)
    y = y.reset_index(drop=True)

    target = pd.concat([label_x, y], axis=1)

    #splitting known and missing target values
    known_y_values = target[target[y.name].notnull()].copy()
    missing_y_values = target[target[y.name].isnull()].copy()
    
    #splitting training testing data
    X_labeled = known_y_values.drop(columns=[y.name])
    y_labeled = known_y_values[y.name]

    #y values with NAN values
    X_unlabeled = missing_y_values.drop(columns=[y.name])
    return X_labeled, y_labeled, X_unlabeled

X_labeled, y_labeled, X_unlabeled = y_missing()


# splitting data into test and train data with missing y values predicted using linearregression
def load_predicted_data():
    xp,yp = data_separation()
    #concating know and missing y values
    y_new = yp.copy()
    y_new = y_new
    
    y_predicted = pd.read_csv("y_predicted.csv").squeeze()
    y_new.loc[X_unlabeled.index] = y_predicted

    xp = encoder(xp)
    return train_test_split(xp,y_new ,test_size=0.3, random_state= 0)
    

# splitting data into test and train data with missing y values dropped
def load_clean_data():
    xc , yc = data_separation("yes")
    
    x_new = encoder(xc)
    yc.dropna(axis=0,inplace=True)
   
    return train_test_split(x_new,yc ,test_size=0.3, random_state= 0)

#splitting data into test and train data with y NAN values inputed using simpleimputer
def load_imputed_data():

    xi , yi = data_separation()
    
    xi = encoder(xi)
    #reshaping yi to a 2d array
    yi_reshaped = yi.values.reshape(-1, 1)
    yi_imputed = imputer(yi_reshaped,'y')
    
    yi_clean = pd.Series(yi_imputed.ravel(), index=yi.index)
    yi_clean.head()
    return train_test_split(xi,yi_clean ,test_size=0.3, random_state= 0)
