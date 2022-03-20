import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.kernel_ridge import KernelRidge


def predict_auc(df,selected_features):
    variable_dummy = pd.DataFrame()
    variable_numeral = pd.DataFrame()
    feature_symbolic = [feature for feature in selected_features if df[feature].dtype == object]
    if len(feature_symbolic) != 0:
        variable_dummy = pd.get_dummies(df[feature_symbolic], dummy_na=True)
    feature_numeral = [feature for feature in selected_features if df[feature].dtype != object]
    if len(feature_numeral) != 0:
        variable_numeral = pd.get_dummies(df[feature_numeral], dummy_na=True)
    if not variable_dummy.empty and not variable_numeral.empty:
        x = pd.concat([variable_numeral, variable_dummy], axis=1)
    elif variable_dummy.empty:
        x = variable_numeral
    else:
        x = variable_dummy
    y = df['Survived']
    # model = LogisticRegression(max_iter=300)
    model = KernelRidge(kernel='rbf')
    model.fit(x, y)
    return model,x


def predict_test(df,selected_features,model,x_train):
    variable_dummy = pd.DataFrame()
    variable_numeral = pd.DataFrame()
    feature_symbolic = [feature for feature in selected_features if df[feature].dtype == object]
    if len(feature_symbolic) != 0:
        variable_dummy = pd.get_dummies(df[feature_symbolic], dummy_na=True)
    feature_numeral = [feature for feature in selected_features if df[feature].dtype != object]
    if len(feature_numeral) != 0:
        variable_numeral = pd.get_dummies(df[feature_numeral], dummy_na=True)
    if not variable_dummy.empty and not variable_numeral.empty:
        x = pd.concat([variable_numeral, variable_dummy], axis=1)
    elif variable_dummy.empty:
        x = variable_numeral
    else:
        x = variable_dummy
    x = x.reindex(labels=x_train.columns, axis=1)
    x=x.fillna(0)
    pred_test = model.predict_proba(x)[:,1]
    return pred_test


def wrapper_method(features):
    selected_features = []
    auc_iteration = []
    for i in range(len(features)):
        print('iteration:',i)
        auc_list=[]
        for column in features:
            variable_dummy=pd.DataFrame()
            variable_numeral=pd.DataFrame()
            if df[column].dtype==object:
                variable_dummy = pd.get_dummies(df[[column] + [feature for feature in selected_features if df[feature].dtype == object]], dummy_na=True)
                feature_numeral=[feature for feature in selected_features if df[feature].dtype != object]
                if len(feature_numeral)!=0:
                    variable_numeral = pd.get_dummies(df[feature_numeral], dummy_na=True)
            else:
                variable_numeral = df[[column] + [feature for feature in selected_features if df[feature].dtype != object]]
                feature_symbolic=[feature for feature in selected_features if df[feature].dtype == object]
                if len(feature_symbolic)!=0:
                    variable_dummy = pd.get_dummies(df[feature_symbolic], dummy_na=True)
                if i==0:
                    variable_numeral=variable_numeral.values.reshape(-1, 1)
            if not variable_dummy.empty and not variable_numeral.empty:
                x=pd.concat([variable_numeral, variable_dummy],axis=1)
            elif variable_dummy.empty:
                x=variable_numeral
            else:
                x=variable_dummy
            y=df['Survived']
            # model=LogisticRegression(max_iter=300)
            model = KernelRidge(kernel='rbf')
            scores = cross_val_score(model, x, y, scoring='roc_auc', cv=10)
            auc_list.append(scores.mean())

        max_feature_index=auc_list.index(max(auc_list))
        selected_features.append(features[max_feature_index])
        print('max AUC:',max(auc_list),'    feature:',selected_features)
        if i != 0:
            if max(auc_iteration) >= max(auc_list):
                print('**** stop iteration ****')
                return selected_features[:-1]
        auc_iteration.append(max(auc_list))
        features=np.delete(features,max_feature_index)
    return selected_features


def _data_shaping(i,used_column,df):
    variable_dummy = pd.DataFrame()
    variable_numeral = pd.DataFrame()
    if df[used_column].dtype == object:
        variable_dummy = pd.get_dummies(
            df[[used_column] + [feature for feature in selected_features if df[feature].dtype == object]],
            dummy_na=True)
        feature_numeral = [feature for feature in selected_features if df[feature].dtype != object]
        if len(feature_numeral) != 0:
            variable_numeral = pd.get_dummies(df[feature_numeral], dummy_na=True)
    else:
        variable_numeral = df[
            [used_column] + [feature for feature in selected_features if df[feature].dtype != object]]
        feature_symbolic = [feature for feature in selected_features if df[feature].dtype == object]
        if len(feature_symbolic) != 0:
            variable_dummy = pd.get_dummies(df[feature_symbolic], dummy_na=True)
        if i == 0:
            variable_numeral = variable_numeral.values.reshape(-1, 1)
    if not variable_dummy.empty and not variable_numeral.empty:
        x = pd.concat([variable_numeral, variable_dummy], axis=1)
    elif variable_dummy.empty:
        x = variable_numeral
    else:
        x = variable_dummy
    return x


df = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')
df["Age"] = df["Age"].fillna(df["Age"].median())  # 欠損値を埋める
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # 欠損値を埋める
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())  # 欠損値を埋める
df_test["Embarked"] = df_test["Embarked"].fillna(df_test["Embarked"].mode()[0])  # 欠損値を埋める
features= df.columns.values[2:]
features=np.delete(features,1)  # Nameを削除
print('特徴量選択する特徴量',features)
selected_features=wrapper_method(features)
print('Wrapper Methodによって選択された特徴量',selected_features)
model,x_train=predict_auc(df, selected_features)  # 選択された特徴量で再度学習を行う
pred=predict_test(df_test,selected_features,model,x_train)
# 予測したデータを保存
sub_df = pd.DataFrame({
    'target': pred
})
sub_df.to_csv('./submission.csv', index=False)
print('submission.csv is stored')
