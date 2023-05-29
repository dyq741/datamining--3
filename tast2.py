import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

# 1. 数据预处理
# 读取数据集
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 删除缺失数据
data = data.dropna()

# 将数据类型转换为数值类型
# 首先，对非数值型特征进行编码
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# 2. 特征选择
# 使用相关性分析选择特征
correlation_matrix = data.corr()
correlation_with_target = abs(correlation_matrix['Churn']).sort_values(ascending=False)
selected_features = correlation_with_target[correlation_with_target > 0.1].index.tolist()

# 使用递归特征消除选择特征
X = data.drop('Churn', axis=1)
y = data['Churn']
selector = SelectKBest(score_func=chi2, k=5)
selector.fit(X, y)
selected_features = X.columns[selector.get_support()].tolist()

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data[selected_features], data['Churn'], test_size=0.3, random_state=42)

# 4. 构建决策树模型
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 5. 模型评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
