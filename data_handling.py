import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv('Melbourne_housing_FULL.csv')
data.head()

data.drop(["Lattitude", "Longtitude",'Address', 'Method', 'SellerG', 'Date', 'Postcode', 'Regionname', 'Propertycount'],axis="columns", inplace=True)

data.head(10)

#removiving target column from training data
train = data.copy()
train.drop(['Price'],axis='columns',inplace=True)

# creating X and Y array
x = train
y = data['Price']

# splitting data into test and train data
x_train, x_test ,y_train ,y_test = train_test_split(x,y ,test_size=0.3, random_state= 0)

#looking for missing x values
missing_values = [col for col in x.columns
                   if x[col].isnull().any()]


print(missing_values)
x.head()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')

#inputation
imputed_x_train = pd.DataFrame(imputer.fit_transform(x_train))
imputed_x_test = pd.DataFrame(imputer.transform(x_test))

# Imputation removed column names; put them back
imputed_x_train.columns = x.columns
imputed_x_test.columns = x_test.columns


imputed_x_train.head()

#getting list of categorical data

missing_dtypes = [cols for cols in x_train.columns
          if x_train[cols].dtype == "object"]

from sklearn.preprocessing import OrdinalEncoder

ordinal_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Make copy to avoid changing original data preprocess data
label_x_train = imputed_x_train.copy()
label_x_test = imputed_x_test.copy()

# Apply ordinal encoder to each column with categorical data
label_x_train[missing_dtypes] = ordinal_enc.fit_transform(imputed_x_train[missing_dtypes])
label_x_test[missing_dtypes] = ordinal_enc.transform(imputed_x_test[missing_dtypes])

print(label_x_train.isnull().sum())
label_x_test.head()

#dealing with missing target column values
label_x_train = label_x_train.reset_index(drop=True) 
y_train = y_train.reset_index(drop=True)

target = pd.concat([label_x_train, y_train], axis=1)
print(target.isnull().sum())

#splitting known and missing target values
known_y_values = target[target[y_train.name].notnull()]
missing_y_values = target[target[y_train.name].isnull()]


print(y_train.isnull().sum(), " length: " , len(y_train), "known: " , known_y_values.isnull().sum(), "unknown: " , missing_y_values.isnull().sum())