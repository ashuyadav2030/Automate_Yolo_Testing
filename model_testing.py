import os
import requests
import yaml
import pandas as pd
from collections import Counter
import time

# ---------------- CONFIG ------------------
LABEL_DIR = r"/mnt/c/Users/ashuy/Documents/face/test/labels"
IMAGE_DIR = r"/mnt/c/Users/ashuy/Documents/face/test/images"
YAML_PATH = r"/mnt/c/Users/ashuy/Documents/face/test/data.yaml"
API_URL = "YOUR_API_URL"
OUTPUT_EXCEL = os.path.join(IMAGE_DIR, "Waf_Detection_Report_Final.xlsx")

HEADERS = {"accept": "application/json"}
IGNORE_KEYS = {
    "width", "latest_date", "height", "model", "store_id", "user_id", "type_name",
    "brightness", "blur", "all_products", "img_url", "all_products_image_url",
    "json_url", "unique_id"
}
# ------------------------------------------


def load_class_map(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if isinstance(data["names"], list):
        return {str(i): name for i, name in enumerate(data["names"])}
    else:
        return {str(k): v for k, v in data["names"].items()}


def load_ground_truth_counts(label_file, class_map):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    class_ids = [line.strip().split()[0] for line in lines]
    return Counter([class_map.get(cls_id, f"class_{cls_id}") for cls_id in class_ids])


def get_prediction(image_path):
    print(f"\n📤 Sending: {image_path}")
    with open(image_path, 'rb') as f:
        files = {'files': (os.path.basename(image_path), f, 'image/jpeg')}
        try:
            response = requests.post(f"{API_URL}&ts={int(time.time())}", headers=HEADERS, files=files)
        except Exception as e:
            print(f"❌ Request error: {e}")
            return None

    print(f"📥 Status: {response.status_code}")
    if response.status_code == 200:
        try:
            return response.json()
        except Exception as e:
            print(f"❌ JSON decode error: {e}")
            return None
    else:
        print(f"❌ Failed: {image_path} - {response.status_code}")
        return None


def replace_ext(filename):
    return os.path.splitext(filename)[0] + ".txt"


def generate_report():
    class_map = load_class_map(YAML_PATH)
    excel_data = []

    for file in os.listdir(IMAGE_DIR):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".jfif", ".png")):
            filename = file
            image_path = os.path.join(IMAGE_DIR, filename)
            label_path = os.path.join(LABEL_DIR, replace_ext(filename))

            if not os.path.exists(label_path):
                print(f"⚠️ Skipping: {filename} (no label)")
                continue

            gt_counts = load_ground_truth_counts(label_path, class_map)
            print(f"\n📸 Processing: {filename}")
            print(f"🟩 Ground Truth Counts: {dict(gt_counts)}")

            response_json = get_prediction(image_path)
            if not response_json:
                continue

            output_image_link = response_json.get("images", [{}])[0].get("output_image", "N/A")

            # ✅ Extract prediction counts from result[0] and classify
            pred_counts = {}
            britannia_total = 0
            competition_total = 0
            result = response_json.get("result", [])
            if result and isinstance(result, list):
                result_data = result[0]
                for key, value in result_data.items():
                    if key in IGNORE_KEYS:
                        continue
                    count = int(float(value))
                    pred_counts[key] = count
                    if key.upper().startswith("BRITANNIA"):
                        britannia_total += count
                    else:
                        competition_total += count

            total = britannia_total + competition_total
            britannia_Share = round((britannia_total / total) * 100, 2) if total > 0 else 0
            competition_Share = round((competition_total / total) * 100, 2) if total > 0 else 0

            print(f"✅ Predicted counts: {pred_counts}")

            all_products = set(gt_counts.keys()).union(pred_counts.keys())

            for product_name in all_products:
                manual = gt_counts.get(product_name, 0)
                model = pred_counts.get(product_name, 0)
                diff = abs(manual - model)
                error_rate = round(diff / manual, 2) if manual else (1.0 if model > 0 else 0.0)

                if manual == 0 and model > 0:
                    status = "False Positive"
                elif manual > 0 and model == 0:
                    status = "False Negative"
                elif manual == model:
                    status = "Match"
                else:
                    status = "Mismatch"

                print(f"🔁 {product_name} | GT: {manual} | Model: {model} | Status: {status}")

                excel_data.append({
                    "Image Name": filename,
                    "Output Image Link": output_image_link,
                    "Product Name": product_name,
                    "Manual Count": manual,
                    "Model Count": model,
                    "Difference": diff,
                    "Error Rate": error_rate,
                    "Detection Status": status,
                    "Britannia Share": britannia_Share,
                    "Competition Share": competition_Share
                })

    # Save to Excel
    df = pd.DataFrame(excel_data)
    try:
        df.to_excel(OUTPUT_EXCEL, index=False)
        print(f"\n✅ Report Saved: {OUTPUT_EXCEL}")
    except Exception as e:
        print(f"❌ Error saving Excel: {e}")


# Run
if __name__ == "__main__":
    generate_report()
