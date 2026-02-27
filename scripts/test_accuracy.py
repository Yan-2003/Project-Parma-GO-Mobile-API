from PreTrain import preprocess, blur_score, run_ocr, correct_drug, lang_score, CONF_THRESHOLD, BLUR_THRESHOLD
from PIL import Image
import os, csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "Testing")
CSV_FILE = os.path.join(TEST_DIR, "testing_labels.csv")
IMAGES_DIR = os.path.join(TEST_DIR, "testing_words")

def predict(image_path):
    img = preprocess(Image.open(image_path))
    clarity = blur_score(img)
    if clarity < BLUR_THRESHOLD:
        return "", "blurry", None
    results = run_ocr(img)
    best = None
    for text, conf in results:
        if not text or not any(c.isalpha() for c in text): continue
        if conf < CONF_THRESHOLD: continue
        corr, match_score = correct_drug(text)
        if not corr: continue
        lscore = lang_score(corr)
        final_score = conf*0.6 + match_score*0.3 + lscore*0.1
        if best is None or final_score > best[1]:
            best = (corr, final_score)
    if best:
        return best[0], "ok", best
    else:
        return "", "low_conf", None

def main():
    total = correct = blurry = low_conf = 0
    mismatches = []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            imgname = row["IMAGE"].strip()
            true_name = row["MEDICINE_NAME"].strip()
            path = os.path.join(IMAGES_DIR, imgname)
            if not os.path.exists(path): continue
            pred, status, _ = predict(path)
            total += 1
            if status=="blurry": blurry += 1
            elif status=="low_conf": low_conf += 1
            elif status=="ok":
                if pred.lower()==true_name.lower(): correct+=1
                else: mismatches.append((imgname,true_name,pred))
    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total*100:.2f}%, Blurry: {blurry}, Low Conf: {low_conf}, Mismatches: {len(mismatches)}")

if __name__=="__main__":
    main()