
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.linear_model import LinearRegression
from data_handling import x,y,x_train,x_test,y_test,y_train, known_y_values,missing_y_values

#training target model for missing values

# Separate inputs and target
X_labeled = known_y_values.drop(columns=[y_train.name])
y_labeled = known_y_values[y_train.name]




# Train a model
reg = LinearRegression()
reg.fit(X_labeled, y_labeled)

X_unlabeled = missing_y_values.drop(columns=[y_train.name])
y_predicted = reg.predict(X_unlabeled)

# Add predicted prices back
missing_y_values[y_train.name] = y_predicted

# setting up model hyperparameters
modele = ensemble.GradientBoostingRegressor(
n_estimators=150,
learning_rate=0.1,
max_depth=30,
min_samples_split=4,
min_samples_leaf=6,
max_features=0.6,
loss='huber'
)
modell = LinearRegression(fit_intercept=True, n_jobs=-1, positive=False)

modelr = ensemble.RandomForestRegressor(n_estimators=100,random_state=0)

modelg = ensemble.GradientBoostingRegressor()

models = [modele,modell, modelr, modelg]

def score_test(model,xt = x_train, xte = x_test, yt = y_train, yte = y_test):
    model.fit(xt, yt)
    preds = model.predict(xte)
    return mean_absolute_error(yte, preds)


#checking model accuracy
for i in range(0, len(models)):
    mae = score_test(models[i])
    print(f"Model {i+1} MAE: {mae:.4f}")

#saving model to a file
# joblib.dump(model, 'house_trained_model.pkl')