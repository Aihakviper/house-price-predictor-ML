
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.linear_model import LinearRegression
from data_handling import load_predicted_data,load_clean_data, y_missing,pd , load_imputed_data

#assingning variables
x_train, x_test, y_train, y_test = load_predicted_data()
x_trainr, x_testr ,y_trainr ,y_testr = load_clean_data()
X_labeled, y_labeled, X_unlabeled = y_missing()
xc_train ,xc_test , yc_train, yc_test = load_imputed_data()

#training target model for missing values. i.e training a model on know y(price) values then using that model to predict missing y(price) values.
# y and x and cancat together before the separtion then after dropping y missing column x columns where used on the model to predict the y values.


def train_target_imputer(X_labeled, y_labeled, X_unlabeled):
    model = LinearRegression()
    model.fit(X_labeled, y_labeled)
    y_pred = model.predict(X_unlabeled)
    return y_pred

y_predicted = train_target_imputer(X_labeled, y_labeled, X_unlabeled)

# saving train y missing values to a csv file
pd.Series(y_predicted).to_csv("y_predicted.csv", index=False)


# setting up model hyperparameters

modell = LinearRegression(fit_intercept=True, n_jobs=-1, positive=False)


modelr = ensemble.RandomForestRegressor(n_estimators=100,random_state=0)
modelr2 = ensemble.RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

modelg = ensemble.GradientBoostingRegressor()

# modele,modele2,

models = [ modell, modelr,modelr2, modelg]

def score_test(model,xt , xte , yt , yte ):
    model.fit(xt, yt)
    preds = model.predict(xte)
    return mean_absolute_error(yte, preds)


# #checking model accuracy for predicted  values . i.e y (price) missing values where separated into known and unknown then a model was train
# #using the train model i predicted the unknown values. function name is load_predicted_data in data_handling file.
for i in range(0, len(models)):
    mae = score_test(models[i], x_train, x_test, y_train, y_test)
    print(f"Model {i+1} MAE: {mae:.4f}")

# model accuracy for clean data. i.e data with drop rows of missing values. function name: load_clean_data in data_handling file.
for i in range(0, len(models)):
    mae = score_test(models[i], x_trainr, x_testr, y_trainr, y_testr)
    print(f"Model {i+1} MAE: {mae:.4f}")

# model accuracy for imputed data. i.e data with target missing values(price) imputed using simple imouter located in data handling file.
for i in range(0, len(models)):
    mae = score_test(models[i], xc_train, xc_test, yc_train, yc_test)
    print(f"Model {i+1} MAE: {mae:.4f}")

#saving best performing model....

# best performing model for predicted data 
predicted_model = "pass"


# best performing model for clean data is a random forest model: modelr
clean_model = modelr.fit(x_trainr, y_trainr)

#best performing model for imputed data is random forest model: modelr2.
imputed_model = modelr2.fit(xc_train, yc_train)


#saving model to a file
#saving for predicted data
# joblib.dump(modelr, 'house_trained_model.pkl')
# joblib.dump(x_train.columns.tolist(), "model_features.pkl")

#saving for clean data
joblib.dump(modelr, 'house_trained_clean_model.pkl')
joblib.dump(x_train.columns.tolist(), "clean_model_features.pkl")

#saving for imputed data
joblib.dump(modelr2, 'house_trained_imputed_model.pkl')
joblib.dump(x_train.columns.tolist(), "imputed_model_features.pkl")
