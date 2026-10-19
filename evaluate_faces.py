import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 順便關掉那些紅色的 TensorFlow 煩人警告

import csv
import cv2
from deepface import DeepFace

# 1. 基本路徑與參數設定
original_human = "/home/han/final/Human/08.jpg" 
animals = ["rabbit", "cat", "bird", "bear", "frog"]
base_dir_prefix = "GenerateResult_"
output_csv = "03_deepface_evaluation_results.csv"

# 建立一個專門用來存 MTCNN 裁切臉部圖的資料夾，畫面更整潔
debug_crop_dir = "detected_faces_debug"
os.makedirs(debug_crop_dir, exist_ok=True)

# ==========================================
#  自動縮圖函數 (避免高畫質原圖塞爆記憶體)
# ==========================================
def resize_if_too_large(img, max_dim=1024):
    """如果圖片長或寬超過 max_dim，就等比例縮小"""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h))
    return img

print("🚀 開始執行 IP-Adapter 人臉一致性量化評估 (含 MTCNN 視覺化裁切存檔)\n" + "="*50)

# ==========================================
# 2. 手動強制讀取人類原圖並縮小
# ==========================================
print(f"正在嘗試讀取人類原圖: {original_human}")
img1_array = cv2.imread(original_human)

if img1_array is None:
    print(f"❌ 嚴重錯誤：OpenCV 無法讀取原圖！")
    exit()
else:
    img1_array = resize_if_too_large(img1_array)
    print(f"✅ 原圖讀取且縮放成功！優化後圖片大小為: {img1_array.shape}")

# ==========================================
# 3. 開始進行比對與裁切存檔
# ==========================================
results = []

for animal in animals:
    target_dir = f"{base_dir_prefix}{animal}"
    if not os.path.exists(target_dir):
        continue
        
    print(f"\n📂 開始評估資料夾: {target_dir}")
    
    for filename in sorted(os.listdir(target_dir)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            generated_img_path = os.path.join(target_dir, filename)
            
            img2_array = cv2.imread(generated_img_path)
            if img2_array is None:
                continue
            
            # 生成圖過個水確保安全
            img2_array = resize_if_too_large(img2_array)
            
            try:
                # ==========================================
                # 🎯 【新增功能】：使用 MTCNN 抓臉並存成圖片供簡報使用
                # ==========================================
                detected_faces = DeepFace.extract_faces(
                    img_path=generated_img_path,
                    detector_backend="mtcnn",
                    enforce_detection=True, # True : 如果沒抓到真正的臉就直接報錯跳過 False : 避免動物特徵太強找不到臉而中斷
                    align=True               # 自動五官對齊與轉正
                )
                
                # 將抓出來的臉存檔
                for i, face_obj in enumerate(detected_faces):
                    face_img = face_obj["face"]
                    if face_img.max() <= 1.0:
                        face_img = (face_img * 255).astype("uint8")
                    
                    bgr_face = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    crop_filename = f"{animal}_{filename}"
                    save_path = os.path.join(debug_crop_dir, crop_filename)
                    cv2.imwrite(save_path, bgr_face)

                # ==========================================
                # 📊 執行原本的 ArcFace 數學距離比對
                # ==========================================
                result = DeepFace.verify(
                    img1_path=img1_array,
                    img2_path=img2_array,
                    model_name="ArcFace",
                    detector_backend="retinaface",  # 這裡用 RetinaFace 就好，因為我們已經用 MTCNN 抓過臉了
                    enforce_detection=True
                )
                
                distance = result["distance"]
                verified = result["verified"]
                results.append([animal, filename, f"{distance:.4f}", verified])
                print(f"  [成功] {filename} | 距離差異: {distance:.4f} | 判定同人: {verified} (已存裁切圖)")
                
            except Exception as e:
                error_msg = str(e).replace("\n", " ")
                results.append([animal, filename, "Error", error_msg])
                print(f"  [失敗] {filename} | 錯誤原因: {error_msg}")

# 4. 將數據匯出成 CSV 報表
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Animal_Category", "Filename", "Distance_Score", "Is_Same_Person_Or_Error"])
    writer.writerows(results)

# ==========================================
# 5. 自動統計與總結報告
# ==========================================
print("\n" + "="*50)
print("📊 實驗結果統計報告")
print("="*50)

true_count = 0
true_list = []

for row in results:
    animal_name = row[0]
    filename = row[1]
    is_same = row[3]  
    
    if is_same == True:
        true_count += 1
        true_list.append(f"[{animal_name}] {filename}")

print(f"🎯 總共有 {true_count} 張圖片被系統判定為「同一人」！")
print(f"📂 MTCNN 裁切過的人臉圖已全部儲存至資料夾：`{debug_crop_dir}/`")
print(f"\n✅ 評估完成！數據已儲存至 {output_csv}")