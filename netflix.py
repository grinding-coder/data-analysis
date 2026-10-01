# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"D:\Chat records\QQ Downloads\python数据分析\netflix_user_behavior_churn_50000.csv")

# 目标变量：用户平均打分 avg_rating_given
target = "avg_rating_given"
# 删除缺失样本
df = df.dropna(subset=[target])

# 划分特征、y
y = df[target].copy()
X_raw = df.drop([target, "churned"], axis=1)

# 区分：数值特征、类别特征
num_cols = ["account_age_months", "session_count", "avg_watch_time_minutes_per_week",
            "watch_sessions_per_week", "completion_rate", "recommendation_click_rate",
            "days_since_last_login", "app_rating"]
cat_cols = ["favorite_genre", "time_of_day", "recommendation_source"]

# ==========2.预处理管道 ==========
# 数值特征Z‑score标准化；类别特征One‑Hot编码
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), cat_cols)
    ]
)

# ==========3.划分训练集、测试集 ==========
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.3, random_state=42
)

# ==========4.ElasticNet：10折交叉验证，同时搜索alpha和l1_ratio ==========
# l1_ratio=0 → Ridge；l1_ratio=1 → Lasso；中间混合ElasticNet
enet = ElasticNetCV(
    l1_ratio=[0.1,0.3,0.5,0.7,0.8,0.9,0.95],
    alphas=np.logspace(-4, 2, 100),
    cv=10,
    max_iter=20000,
    random_state=42
)

# 构建完整流水线：预处理+弹性网
pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("elasticnet", enet)
])

pipe.fit(X_train_raw, y_train)

# ==========5.输出结果 ==========
best_alpha = pipe.named_steps["elasticnet"].alpha_
best_l1ratio = pipe.named_steps["elasticnet"].l1_ratio_
print("="*60)
print(f"✅ 最佳alpha(λ): {best_alpha:.6f}")
print(f"✅ 最佳l1_ratio: {best_l1ratio:.3f}")

# 测试集预测
y_pred = pipe.predict(X_test_raw)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ 测试集RMSE: {rmse:.4f}")

# 获取全部特征名称
ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
all_feature_names = num_cols + cat_feature_names

coefs = pipe.named_steps["elasticnet"].coef_
coef_series = pd.Series(coefs, index=all_feature_names)

# 筛选非零系数（ElasticNet实现稀疏）
non_zero_coef = coef_series[np.abs(coef_series) > 1e-5].sort_values(key=abs, ascending=False)
print("\n📌 ElasticNet选出的非零特征（按绝对值排序）：")
print(non_zero_coef)

# ==========6.可视化：系数条形图 ==========
plt.figure(figsize=(10,7))
non_zero_coef.plot(kind="barh")
plt.title("ElasticNet 非零回归系数 Netflix 用户评分预测")
plt.xlabel("回归系数大小")
plt.tight_layout()
plt.show()

