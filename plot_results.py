import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 讀取與過濾數據
# ==========================================
csv_file = "03_deepface_evaluation_results.csv"

# 讀取 CSV
df = pd.read_csv(csv_file)

# 過濾掉 Error 的資料，只保留成功算出身分距離的數字
df_valid = df[df["Distance_Score"] != "Error"].copy()
# 確保距離分數是浮點數格式
df_valid["Distance_Score"] = df_valid["Distance_Score"].astype(float)

# ==========================================
# 2. 設定繪圖風格與畫布
# ==========================================
plt.figure(figsize=(10, 6))
# 設定 seaborn 美化風格
sns.set_theme(style="whitegrid")

# ==========================================
# 3. 繪製直方圖 (Histogram) 與密度曲線 (KDE)
# ==========================================
# 將所有動物的數據一起畫出來看總體分佈
sns.histplot(
    data=df_valid, 
    x="Distance_Score", 
    bins=15, 
    kde=True, 
    color="#4C72B0", 
    edgecolor="black",
    alpha=0.7
)

# ==========================================
# 4. 畫出關鍵的 0.68 閾值紅線
# ==========================================
threshold = 0.68
plt.axvline(x=threshold, color="red", linestyle="--", linewidth=2.5, label=f"ArcFace Threshold (0.68)")

# ==========================================
# 5. 添加圖表標題與標籤
# ==========================================
plt.title("ArcFace Cosine Distance Distribution in Cross-Species Fusion", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Cosine Distance (Lower means more similar)", fontsize=12)
plt.ylabel("Frequency (Number of Images)", fontsize=12)

# 在紅線旁邊加個文字說明
plt.text(threshold - 0.02, plt.ylim()[1]*0.8, 'True (Same Person)\nDistance <= 0.68', 
         color='green', horizontalalignment='right', fontweight='bold')
plt.text(threshold + 0.02, plt.ylim()[1]*0.8, 'False (Different Person)\nDistance > 0.68', 
         color='darkred', horizontalalignment='left', fontweight='bold')

plt.legend()
plt.tight_layout()

# ==========================================
# 6. 儲存圖表並顯示
# ==========================================
output_img = "distance_distribution.png"
plt.savefig(output_img, dpi=300)
print(f"✅ 圖表已成功儲存為: {output_img}")
plt.show()