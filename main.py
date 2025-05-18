import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib

data = pd.read_csv('Melbourne_housing_FULL.csv')

# dropping unwanted coulums
data.drop(["Lattitude", "Longtitude",'Address', 'Method', 'SellerG', 'Date', 'Postcode', 'Regionname', 'Propertycount'],axis="columns", inplace=True)


# drapping null rows / rows with missing data
data.dropna(axis=0, how='any', subset=None, inplace=True)

#one hot encoding, converting non numeriacl to numerical
enc_data = pd.get_dummies(data,columns=['Suburb','Type', 'CouncilArea'])

#removiving target column from training data
enc_data.drop(['Price'],axis='columns',inplace=True)

# creating X and Y arrays
x = enc_data.values
y = data['Price'].values

# print(enc_data.head())
# splitting data into test and train data
x_train, x_test ,y_train ,y_test = train_test_split(x,y ,test_size=0.3, random_state= 0)

# setting up model hyperparameters
model = ensemble.GradientBoostingRegressor(
n_estimators=150,
learning_rate=0.1,
max_depth=30,
min_samples_split=4,
min_samples_leaf=6,
max_features=0.6,
loss='huber'
)

# running model on trained data
model.fit(x_train,y_train)

#saving model to a file
joblib.dump(model, 'house_trained_model.pkl')

#checking model accuracy
mse = mean_absolute_error(y_train, model.predict(x_train))
print ("Training Set Mean Absolute Error: %.2f" % mse)

mse2 = mean_absolute_error(y_test ,model.predict(x_test))
print('Test set Mean absolute Error: %.2f' % mse2)