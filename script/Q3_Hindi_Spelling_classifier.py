import re
import requests
import unicodedata
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from rapidfuzz import process, fuzz

BASE_DIR = Path("/teamspace/studios/this_studio")
DATA_DIR = BASE_DIR / "data"
Q1_OUT = BASE_DIR / "outputs" / "q1"
TRANS_DIR = Q1_OUT / "transcriptions"
OUT_DIR = BASE_DIR / "outputs" / "q3"
OUT_DIR.mkdir(exist_ok=True, parents=True)

CSV_PATH = DATA_DIR / "FT Data - data.csv"

DEVANAGARI_LOANWORDS = {
    'कंप्यूटर', 'कम्प्यूटर', 'इंटरनेट', 'मोबाइल', 'फोन', 'वीडियो',
    'ऑनलाइन', 'सॉफ्टवेयर', 'हार्डवेयर', 'ऐप', 'वेबसाइट', 'ईमेल',
    'स्क्रीन', 'कीबोर्ड', 'माउस', 'लैपटॉप', 'टैबलेट', 'स्मार्टफोन',
    'इंटरव्यू', 'जॉब', 'ऑफिस', 'मैनेजर', 'टीम', 'प्रोजेक्ट',
    'मीटिंग', 'प्रेजेंटेशन', 'रिपोर्ट', 'टारगेट', 'बजट',
    'बैंक', 'लोन', 'क्रेडिट', 'डेबिट', 'स्टॉक',
    'प्रॉब्लम', 'सॉल्व', 'मैनेज', 'अपडेट', 'डाउनलोड',
    'नॉर्मल', 'रेगुलर', 'स्पेशल', 'फाइनल', 'टोटल',
    'मूवी', 'सीरीज', 'चैनल', 'न्यूज़', 'न्यूज',
    'सोशल', 'मीडिया', 'यूट्यूब', 'इंस्टाग्राम', 'फेसबुक',
    'डॉक्टर', 'हॉस्पिटल', 'वैक्सीन', 'बिजनेस', 'स्टार्टअप',
    'मार्केट', 'प्रोडक्ट', 'कस्टमर', 'ट्रेनिंग', 'कोचिंग',
    'एग्जाम', 'रिजल्ट', 'परसेंट', 'फीस', 'बास्केटबॉल',
    'फुटबॉल', 'क्रिकेट', 'कैफेटेरिया', 'इंडस्ट्रीज',
    'गूगल', 'अमेज़न', 'फ्लिपकार्ट', 'लाइव', 'कंटेंट',
    'वायरल', 'ट्रेंड', 'फॉलो', 'लाइक', 'शेयर', 'कमेंट',
    'डिग्री', 'सर्टिफिकेट', 'वर्कशॉप', 'सेमिनार', 'वेबिनार',
    'इमरजेंसी', 'एम्बुलेंस', 'ऑपरेशन', 'इन्वेस्टमेंट',
    'पिच', 'इन्वेस्टर', 'फंडिंग', 'ब्रांड', 'सर्विस',
}


def fix_gcp_url(url):
    match = re.search(r'hq_data/hi/(\d+)/(\d+)_(.*)', url)
    if match:
        folder_id, recording_id, suffix = match.groups()
        return f'https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_{suffix}'
    return url


def fetch_transcription(url, timeout=30):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return " ".join([s["text"].strip() for s in data if s.get("text", "").strip()])
        if isinstance(data, dict):
            return data.get("text") or data.get("transcription", "")
        return None
    except:
        return None


def is_valid_devanagari_structure(word):
    if not word:
        return False, "empty"
    for ch in word:
        if ord(ch) < 0x0900 or ord(ch) > 0x097F:
            if unicodedata.category(ch) not in ('Lo', 'Mn', 'Mc', 'Nd'):
                return False, f"non-Devanagari char: {ch}"
    if re.search(r'(.)\1{2,}', word):
        return False, "repeated chars"
    matras = set('\u093e\u093f\u0940\u0941\u0942\u0943\u0944\u0945\u0946\u0947\u0948\u0949\u094a\u094b\u094c')
    if word[0] in matras:
        return False, "starts with matra"
    if re.search(r'[\u093e-\u094c][\u093e-\u094c]', word):
        return False, "consecutive matras"
    if word.endswith('\u094d'):
        return False, "ends with halant"
    return True, "valid structure"


def classify_word(word, reference_vocab, word_freq):
    word = unicodedata.normalize('NFC', word)
    if word in reference_vocab:
        return "correct", "high", "found in reference vocabulary"
    is_valid, struct_reason = is_valid_devanagari_structure(word)
    if not is_valid:
        return "incorrect", "high", f"invalid structure: {struct_reason}"
    freq = word_freq.get(word, 0)
    if freq >= 5:
        return "correct", "high", f"high frequency in dataset (n={freq})"
    if freq >= 2:
        return "correct", "medium", f"medium frequency in dataset (n={freq})"
    if len(word) <= 1:
        return "correct", "medium", "single character"
    if len(word) > 20:
        return "incorrect", "medium", "unusually long word"
    vocab_sample = list(reference_vocab)[:5000]
    matches = process.extractOne(word, vocab_sample, scorer=fuzz.ratio)
    if matches and matches[1] >= 90:
        return "correct", "medium", f"close match to '{matches[0]}' (similarity={matches[1]})"
    elif matches and matches[1] >= 75:
        return "incorrect", "medium", f"possible typo of '{matches[0]}' (similarity={matches[1]})"
    return "incorrect", "low", f"no close match found in vocabulary (best={matches[1] if matches else 0})"


df = pd.read_csv(CSV_PATH)
df['transcription_url_gcp'] = df['transcription_url_gcp'].apply(fix_gcp_url)

all_words = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting words"):
    rec_id   = str(row['recording_id'])
    txt_path = TRANS_DIR / f"{rec_id}.txt"
    if txt_path.exists():
        text = txt_path.read_text(encoding='utf-8')
    else:
        text = fetch_transcription(row['transcription_url_gcp']) or ""
    all_words.extend(re.findall(r'[\u0900-\u097F]+', text))

word_counts = Counter(all_words)
unique_words = list(word_counts.keys())
dataset_vocab = {w for w, c in word_counts.items() if c >= 3}
reference_vocab = DEVANAGARI_LOANWORDS | dataset_vocab

print(f"Total tokens: {len(all_words):,}")
print(f"Unique words: {len(unique_words):,}")
print(f"Reference vocab: {len(reference_vocab):,}")

results = []
for word in tqdm(unique_words, desc="Classifying"):
    label, confidence, reason = classify_word(word, reference_vocab, word_counts)
    results.append({"word": word, "label": label, "confidence": confidence, "reason": reason, "frequency": word_counts.get(word, 0)})

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "spelling_classification.csv", index=False)

correct = results_df[results_df['label'] == 'correct']
incorrect = results_df[results_df['label'] == 'incorrect']
print(f"Correctly spelled:   {len(correct):,} ({len(correct)/len(results_df)*100:.1f}%)")
print(f"Incorrectly spelled: {len(incorrect):,} ({len(incorrect)/len(results_df)*100:.1f}%)")

low_conf = results_df[results_df['confidence'] == 'low'].copy()
low_conf.sample(min(45, len(low_conf)), random_state=42).to_csv(OUT_DIR / "low_confidence_review.csv", index=False)

final_df = results_df.rename(columns={
    'label':      'spelling_status',
    'confidence': 'confidence_score',
    'reason':     'classification_reason',
}).copy()
final_df['spelling_status'] = final_df['spelling_status'].map({'correct': 'correct spelling', 'incorrect': 'incorrect spelling'})
final_df.sort_values(['spelling_status', 'frequency'], ascending=[True, False]).reset_index(drop=True).to_csv(OUT_DIR / "final_word_classification.csv", index=False)

print(f"Outputs saved to: {OUT_DIR}")