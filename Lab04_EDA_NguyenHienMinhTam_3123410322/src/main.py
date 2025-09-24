# ==============================
# 1. Load dataset & kiểm tra thông tin
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


sns.set(style="whitegrid")

# Đọc dữ liệu
file_path = "pima-indians-diabetes.csv"  
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(file_path, header=None)
df.columns = columns

print("Kích thước dữ liệu:", df.shape)
print(df.info())
df.head()

# ==============================
# 2. Kiểm tra giá trị thiếu
# ==============================
zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("Số lượng giá trị 0 (coi là missing):")
for c in zero_as_missing_cols:
    print(f"{c}: {(df[c]==0).sum()}")

# Thay 0 bằng NaN
df2 = df.copy()
for c in zero_as_missing_cols:
    df2[c] = df2[c].replace(0, np.nan)

print("\nMissing values sau khi thay 0 -> NaN:")
print(df2.isnull().sum())

df2.describe()

# ==============================
# 3. Phân tích đơn biến (Histogram)
# ==============================
df2.hist(bins=20, figsize=(12,10))
plt.suptitle("Histograms (zeros -> NaN)", fontsize=16)
plt.show()

# ==============================
# 4. Phân tích đa biến
# ==============================

# Boxplot theo Outcome
plt.figure(figsize=(14,10))
num_cols = [c for c in df2.columns if c!="Outcome"]
for i, c in enumerate(num_cols, 1):
    plt.subplot(3,3,i)
    sns.boxplot(data=df2, x="Outcome", y=c)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df2.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap")
plt.show()

# ==============================
# 5. Phát hiện ngoại lệ (IQR)
# ==============================
def detect_outliers_iqr(series):
    s = series.dropna()
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers = series[(series < lower) | (series > upper)]
    return len(outliers), lower, upper

for c in num_cols:
    n_out, lo, up = detect_outliers_iqr(df2[c])
    print(f"{c}: {n_out} ngoại lệ (ngưỡng < {lo:.2f} hoặc > {up:.2f})")

# ==============================
# 6. Xử lý missing (imputation)
# ==============================
df_imputed = df2.copy()
for c in zero_as_missing_cols:
    median_val = df_imputed[c].median()
    df_imputed[c] = df_imputed[c].fillna(median_val)
    print(f"Imputed {c} with median = {median_val:.2f}")

print("\nMissing values sau imputation:")
print(df_imputed.isnull().sum())

# Lưu file mới
df_imputed.to_csv("pima_clean_imputed.csv", index=False)
print("Saved cleaned dataset to pima_clean_imputed.csv")

# ==============================
# 7. Phân bố Outcome
# ==============================
plt.figure(figsize=(5,4))
sns.countplot(data=df2, x="Outcome", palette="Set2")
plt.title("Phân bố Outcome (0 = Không ĐTĐ, 1 = ĐTĐ)", fontsize=14)
plt.show()

# ==============================
# 8. So sánh phân phối Glucose và BMI theo Outcome
# ==============================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.kdeplot(data=df2, x="Glucose", hue="Outcome", fill=True, common_norm=False, alpha=0.5)
plt.title("Phân phối Glucose theo Outcome")

plt.subplot(1,2,2)
sns.kdeplot(data=df2, x="BMI", hue="Outcome", fill=True, common_norm=False, alpha=0.5)
plt.title("Phân phối BMI theo Outcome")

plt.tight_layout()
plt.show()

# ==============================
# 9. Scatter plot Glucose vs Age theo Outcome
# ==============================
plt.figure(figsize=(7,5))
sns.scatterplot(data=df2, x="Age", y="Glucose", hue="Outcome", alpha=0.7)
plt.title("Scatterplot Age vs Glucose theo Outcome")
plt.show()

# ==============================
# 10. Pairplot cho vài biến chính
# ==============================
sns.pairplot(df2[["Glucose", "BMI", "Age", "Outcome"]], hue="Outcome", diag_kind="kde", palette="Set1")
plt.suptitle("Pairplot biến chính theo Outcome", y=1.02, fontsize=14)
plt.show()

# 11. Phát hiện mẫu bất thường (Anomaly Detection)
# ==============================
# Isolation Forest (đa biến)
iso = IsolationForest(contamination=0.05, random_state=42)
df_imputed["anomaly"] = iso.fit_predict(df_imputed.drop(columns=["Outcome"]))

print("\nKết quả phát hiện anomaly:")
print(df_imputed["anomaly"].value_counts())  # 1 = normal, -1 = anomaly

print("\nMột vài mẫu bất thường được phát hiện:")
print(df_imputed[df_imputed["anomaly"]==-1].head())

# Trực quan hóa anomalies (Glucose vs BMI)
plt.figure(figsize=(7,5))
sns.scatterplot(data=df_imputed, x="Glucose", y="BMI",
                hue="anomaly", palette={1:"blue", -1:"red"}, alpha=0.7)
plt.title("Isolation Forest - Phát hiện mẫu bất thường (Glucose vs BMI)")
plt.show()