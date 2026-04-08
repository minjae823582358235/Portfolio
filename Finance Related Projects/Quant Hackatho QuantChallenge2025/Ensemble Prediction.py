#%% Y2 Prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import joblib
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('train.csv')
df=df[['Y1','B','D','F','I','K','L','Y2']]

maxLag=5
target='Y2'

# Creating lagging features

df_lagged=df.copy()
for lag in range(1,maxLag+1):
    for col in df.columns:
        df_lagged[f"{col}_lag{lag}"]=df[col].shift(lag)
df_lagged.dropna(inplace=True)
df_lagged.drop('Y1',axis=1,inplace=True)
feature_cols=[c for c in df_lagged.columns if c is not 'Y2']
print(df_lagged)
X=df_lagged[feature_cols]
y=df_lagged[target]

trainsize=int(0.8*len(df_lagged))
X_train, X_test=X.iloc[:trainsize],X.iloc[trainsize:]
y_train, y_test=y.iloc[:trainsize],y.iloc[trainsize:]


nTrees=500
rf= RandomForestRegressor(
    n_estimators=1,
    max_depth=10,
    warm_start=True,
    random_state=8235
)
for n in tqdm(range(1,nTrees+1)):
    rf.n_estimators=n
    rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

r2=r2_score(y_test,y_pred)
print("R2:", r2)

plt.figure(figsize=(14,4))
plt.plot(y_test.index,y_test,label='Actual Y2',color='tab:blue')
plt.plot(y_test.index, y_pred, label='RF Predicted Y2',color='tab:orange',linestyle='--')
plt.legend()
plt.title(f"Random Forest Prediction of Y2 (R2={r2:.4f})")
plt.show()
importances=rf.feature_importances_
feat_importance=pd.Series(importances, index=feature_cols).sort_values(ascending=False)
print("Top features:/n", feat_importance.head(20))
joblib.dump(rf,'Y2RandomForestY2Tau1OmitY1TauRawDog.pkl')

#%%
max_lag=2
N=2
df=pd.read_csv('train.csv')

features= [col for col in df.columns if col not in ['Y1','Y2','time']]
X_features = ['time']

for col in features:
    X_features.append(col)
    for lag_window in range(1, max_lag +1):
        df[f'{col}_lag_{lag_window}']=df[col].shift(lag_window).fillna(0)
        X_features.append(f'{col}_lag_{lag_window}')
    
    df[f'{col}_ewmstd_{N}']=df[col].rolling(window=N).std().fillna(0)
    X_features.append(f'{col}_ewmstd_{N}')

X = df[X_features]
y=df[['Y1']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=42,shuffle=False)
model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 score Y2:", r2_score(y_test['Y1'], y_pred))
model_preds = model.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['Y1'], label='Actual Y1', color='blue')
plt.plot(df['time'], model_preds, label='Predicted Y1', color='orange')
plt.xlabel('Time')
plt.ylabel('Y1')
plt.title('Actual vs Predicted Y1')
plt.legend()
plt.grid(True)
plt.show()

test_df = pd.read_csv('train.csv')

features = [col for col in test_df.columns if col not in ['Y1', 'Y2', 'time', 'id']]
X_features = ['time']

for col in features:
    X_features.append(col)
    for lag_window in range(1, max_lag + 1):
        test_df[f'{col}_lag_{lag_window}'] = test_df[col].shift(lag_window).fillna(0)
        X_features.append(f'{col}_lag_{lag_window}')
        
    test_df[f'{col}_ewmstd_{N}'] = test_df[col].rolling(window=N).std().fillna(0)
    X_features.append(f'{col}_ewmstd_{N}')

preds = model.predict(test_df[X_features])

y1_csv = pd.DataFrame({
    'id': test_df['id'],
    'Y1': preds
})

y1_csv.to_csv('y1.csv', index=False)