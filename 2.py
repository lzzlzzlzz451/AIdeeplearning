import pandas as pd
from sklearn.model_selection import train_test_split
import feature_engineering.feature_engineering.feature_selection.embedded_method as em

path = r'MachineLearningCVE/resample.csv'
data = pd.read_csv(path)
data_set = data.values
num_samples = len(data)
# 将数据拆分为特征和标签
features = data_set[:, :-1].astype(float)  # 获取除了最后一列的所有列作为特征
labels = data_set[:, -1].astype(int)  # 获取最后一列作为标签
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train = pd.DataFrame(X_train)
Y_train = pd.DataFrame(Y_train)


features_selected = em.rf_importance(X_train,Y_train)
features_selected = features_selected.tolist()
features_selected += [data.shape[1]-1]

filtered_data = data.iloc[:,features_selected]

filtered_data.to_csv('MachineLearningCVE/filtered_data.csv', index=False)



