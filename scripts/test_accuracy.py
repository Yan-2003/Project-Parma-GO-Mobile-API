from PreTrain import preprocess, blur_score, run_ocr, match_drug, BLUR_THRESHOLD
from PIL import Image
import os
import csv
import logging
from datetime import datetime

# =============================
# PATHS
# =============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(BASE_DIR, "Testing")
IMAGES_DIR = os.path.join(TEST_DIR, "testing_words")
CSV_FILE = os.path.join(TEST_DIR, "testing_labels.csv")

LOG_DIR = os.path.join(TEST_DIR, "logs")
RESULT_DIR = os.path.join(TEST_DIR, "results")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================
# TIMESTAMP
# =============================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

LOG_FILE = os.path.join(LOG_DIR, f"ocr_test_{timestamp}.log")
MISMATCH_FILE = os.path.join(RESULT_DIR, f"mismatches_{timestamp}.csv")
SUMMARY_FILE = os.path.join(RESULT_DIR, f"summary_{timestamp}.txt")

# =============================
# LOGGER
# =============================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

print(f"\nLogging to: {LOG_FILE}")

# =============================
# OCR PREDICTION
# =============================

def predict(image_path):

    img = preprocess(Image.open(image_path))

    clarity = blur_score(img)

    if clarity < BLUR_THRESHOLD:
        return "", "blurry"

    ocr_results = run_ocr(img)

    best_match = match_drug(ocr_results)

    if best_match:
        return best_match, "ok"
    else:
        return "", "low_conf"


# =============================
# MAIN TEST
# =============================

def main():

    total = 0
    correct = 0
    blurry = 0
    low_conf = 0

    mismatches = []

    logging.info("===== OCR TEST STARTED =====")

    with open(CSV_FILE, newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            imgname = row["IMAGE"].strip()
            true_name = row["MEDICINE_NAME"].strip()

            image_path = os.path.join(IMAGES_DIR, imgname)

            if not os.path.exists(image_path):
                logging.warning(f"Missing image: {imgname}")
                continue

            pred, status = predict(image_path)

            total += 1

            logging.info(f"IMAGE={imgname} | TRUE={true_name} | PRED={pred} | STATUS={status}")

            if status == "blurry":
                blurry += 1
                continue

            if status == "low_conf":
                low_conf += 1
                continue

            if pred.lower() == true_name.lower():
                correct += 1
            else:
                mismatches.append((imgname, true_name, pred))

    # =============================
    # CALCULATE RESULTS
    # =============================

    accuracy = (correct / total * 100) if total > 0 else 0

    results_text = f"""
========== OCR TEST RESULTS ==========
Total Images: {total}
Correct Predictions: {correct}
Accuracy: {accuracy:.2f}%
Blurry Images: {blurry}
Low Confidence: {low_conf}
Mismatches: {len(mismatches)}
"""

    print(results_text)

    logging.info(results_text)

    # =============================
    # SAVE SUMMARY
    # =============================

    with open(SUMMARY_FILE, "w") as f:
        f.write(results_text)

    # =============================
    # SAVE MISMATCHES
    # =============================

    if mismatches:

        with open(MISMATCH_FILE, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(["IMAGE", "TRUE_LABEL", "PREDICTION"])

            for m in mismatches:
                writer.writerow(m)

        print(f"\nMismatch file saved to: {MISMATCH_FILE}")

    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Log saved to: {LOG_FILE}")

    logging.info("===== OCR TEST FINISHED =====")


# =============================
# RUN
# =============================

if __name__ == "__main__":
    main()