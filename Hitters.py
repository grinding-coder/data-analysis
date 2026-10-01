import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r"D:\Chat records\QQ Downloads\python数据分析\hitters.csv")

# 查看数据基本信息
print(df.shape)
print(df.info())

# 数据预处理：删除Salary缺失样本
df = df.dropna(subset=["Salary"])
# 处理类别变量League、Division、NewLeague，转哑变量
df = pd.get_dummies(df, drop_first=True)

# 划分特征X，目标y
y = df["Salary"]
X = df.drop("Salary", axis=1)

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Z‑score标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 模型定义，10折CV寻找最优alpha
# Ridge（L2正则）
ridge = RidgeCV(alphas=np.logspace(-3, 5, 100), cv=10)
ridge.fit(X_train_scaled, y_train)

# Lasso（L1正则，产生稀疏系数）
lasso = LassoCV(alphas=np.logspace(-3,5,100), cv=10, max_iter=20000)
lasso.fit(X_train_scaled, y_train)

# ElasticNet（L1+L2混合）
enet = ElasticNetCV(
    alphas=np.logspace(-3,5,100),
    l1_ratio=[0.1,0.3,0.5,0.7,0.9],
    cv=10, max_iter=20000
)
enet.fit(X_train_scaled, y_train)

# 3. 在测试集预测并计算RMSE
def get_rmse(model, Xtest, ytest):
    y_pred = model.predict(Xtest)
    return np.sqrt(mean_squared_error(ytest, y_pred))

ridge_rmse = get_rmse(ridge, X_test_scaled, y_test)
lasso_rmse = get_rmse(lasso, X_test_scaled, y_test)
enet_rmse = get_rmse(enet, X_test_scaled, y_test)

# 输出结果
print("="*70)
print(f"Ridge最优alpha = {ridge.alpha_:.4f}, 测试RMSE = {ridge_rmse:.2f}")
print(f"Lasso最优alpha = {lasso.alpha_:.4f}, 测试RMSE = {lasso_rmse:.2f}")
print(f"ElasticNet最优alpha = {enet.alpha_:.4f}, l1_ratio={enet.l1_ratio_:.2f}, 测试RMSE={enet_rmse:.2f}")
print("="*70)

# 输出各模型系数
coef_result = pd.DataFrame({
    "Feature": X.columns,
    "Ridge_coef": ridge.coef_,
    "Lasso_coef": lasso.coef_,
    "ElasticNet_coef": enet.coef_
})
print(coef_result)

# 统计Lasso、ElasticNet的非零系数个数
print("\n非零特征数量：")
print(f"Ridge非零系数：{np.sum(np.abs(ridge.coef_)>1e-5)}")
print(f"Lasso非零系数：{np.sum(np.abs(lasso.coef_)>1e-5)}")
print(f"ElasticNet非零系数：{np.sum(np.abs(enet.coef_)>1e-5)}")

# 4. 绘制系数路径图
alphas = np.logspace(-3, 5, 100)
# Ridge路径
ridge_coefs = []
for a in alphas:
    m = Ridge(alpha=a)
    m.fit(X_train_scaled,y_train)
    ridge_coefs.append(m.coef_)
# Lasso路径
lasso_coefs = []
for a in alphas:
    m = Lasso(alpha=a, max_iter=20000)
    m.fit(X_train_scaled,y_train)
    lasso_coefs.append(m.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.plot(alphas, ridge_coefs)
plt.xscale("log")
plt.title("Ridge 系数路径")
plt.xlabel(r"$\alpha$")
plt.ylabel("系数值")

plt.subplot(1,2,2)
plt.plot(alphas, lasso_coefs)
plt.xscale("log")
plt.title("Lasso 系数路径")
plt.xlabel(r"$\alpha$")
plt.ylabel("系数值")

plt.tight_layout()
plt.show()
