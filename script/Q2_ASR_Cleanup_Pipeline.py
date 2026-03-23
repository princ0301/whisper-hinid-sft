import re
import requests
import unicodedata
import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
import evaluate

BASE_DIR = Path("/teamspace/studios/this_studio")
DATA_DIR = BASE_DIR / "data"
Q1_OUT = BASE_DIR / "outputs" / "q1"
OUT_DIR = BASE_DIR / "outputs" / "q2"
AUDIO_DIR = Q1_OUT / "audio_16k"
TRANS_DIR = Q1_OUT / "transcriptions"
RAW_ASR_DIR = OUT_DIR / "raw_asr"

for d in [OUT_DIR, RAW_ASR_DIR]:
    d.mkdir(exist_ok=True, parents=True)

CSV_PATH  = DATA_DIR / "FT Data - data.csv"
TARGET_SR = 16000

UNITS = {
    'शून्य': 0, 'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4,
    'पाँच': 5, 'पांच': 5, 'छह': 6, 'छः': 6, 'सात': 7,
    'आठ': 8, 'नौ': 9, 'दस': 10, 'ग्यारह': 11, 'बारह': 12,
    'तेरह': 13, 'चौदह': 14, 'पंद्रह': 15, 'सोलह': 16,
    'सत्रह': 17, 'अठारह': 18, 'उन्नीस': 19, 'बीस': 20,
    'इक्कीस': 21, 'बाईस': 22, 'तेईस': 23, 'चौबीस': 24,
    'पच्चीस': 25, 'छब्बीस': 26, 'सत्ताईस': 27, 'अट्ठाईस': 28,
    'उनतीस': 29, 'तीस': 30, 'चालीस': 40, 'पचास': 50,
    'साठ': 60, 'सत्तर': 70, 'अस्सी': 80, 'नब्बे': 90,
    'सौ': 100, 'हज़ार': 1000, 'हजार': 1000,
    'लाख': 100000, 'करोड़': 10000000,
}

MULTIPLIERS = {
    'सौ': 100, 'हज़ार': 1000, 'हजार': 1000,
    'लाख': 100000, 'करोड़': 10000000,
}

IDIOM_PATTERNS = [
    r'दो-चार', r'चार-पाँच', r'दो टूक', r'तीन-तेरह',
    r'नौ-दो ग्यारह', r'एक न एक', r'एक-दो', r'चार-छह',
]

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


def words_to_number(words):
    total, current = 0, 0
    for word in words:
        if word in MULTIPLIERS:
            mult = MULTIPLIERS[word]
            if mult >= 1000:
                total = (total + current) * mult if current else total * mult + mult
                current = 0
            else:
                current = current * mult if current else mult
        elif word in UNITS:
            current += UNITS[word]
    return total + current


def normalize_numbers(text):
    protected, protect_idx = {}, 0
    protected_text = text
    for pattern in IDIOM_PATTERNS:
        for m in re.finditer(pattern, protected_text):
            placeholder = f"__IDIOM{protect_idx}__"
            protected[placeholder] = m.group()
            protected_text = protected_text[:m.start()] + placeholder + protected_text[m.end():]
            protect_idx += 1
    all_nums = sorted(list(UNITS.keys()) + list(MULTIPLIERS.keys()), key=len, reverse=True)
    num_pattern = '|'.join(re.escape(w) for w in all_nums)
    full_pattern = rf'(?:{num_pattern})(?:\s+(?:{num_pattern}))*'

    def replace_seq(match):
        num = words_to_number(match.group().split())
        return str(num) if num != 0 else match.group()

    result = re.sub(full_pattern, replace_seq, protected_text)
    for placeholder, original in protected.items():
        result = result.replace(placeholder, original)
    return result


def detect_english_words(text):
    words = text.split()
    tagged, en_words = [], []
    i = 0
    while i < len(words):
        word       = words[i]
        clean_word = re.sub(r'[^\u0900-\u097Fa-zA-Z]', '', word)
        if re.match(r'^[a-zA-Z]+$', clean_word):
            tagged.append(f"[EN]{word}[/EN]")
            en_words.append(word)
        elif re.search(r'[a-zA-Z]', word) and re.search(r'[\u0900-\u097F]', word):
            tagged.append(f"[EN]{word}[/EN]")
            en_words.append(word)
        elif clean_word in DEVANAGARI_LOANWORDS:
            tagged.append(f"[EN]{word}[/EN]")
            en_words.append(word)
        elif i + 1 < len(words):
            two_word = clean_word + ' ' + re.sub(r'[^\u0900-\u097F]', '', words[i + 1])
            if two_word in DEVANAGARI_LOANWORDS:
                tagged.append(f"[EN]{word} {words[i + 1]}[/EN]")
                en_words.append(f"{word} {words[i + 1]}")
                i += 2
                continue
            else:
                tagged.append(word)
        else:
            tagged.append(word)
        i += 1
    return " ".join(tagged), en_words


def full_pipeline(text):
    normalized = normalize_numbers(text)
    tagged, en_words = detect_english_words(normalized)
    return {"original": text, "normalized": normalized, "tagged": tagged, "en_words": en_words}


df = pd.read_csv(CSV_PATH)
df['transcription_url_gcp'] = df['transcription_url_gcp'].apply(fix_gcp_url)
df['rec_url_gcp']            = df['rec_url_gcp'].apply(fix_gcp_url)
print(f"Recordings: {len(df)}")

valid_ids, references = [], {}

for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading audio + references"):
    rec_id = str(row['recording_id'])
    dst    = AUDIO_DIR / f"{rec_id}.wav"
    if not dst.exists():
        try:
            tmp = AUDIO_DIR / f"{rec_id}_tmp.wav"
            r   = requests.get(row['rec_url_gcp'], timeout=120, stream=True)
            r.raise_for_status()
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            audio, _ = librosa.load(str(tmp), sr=TARGET_SR, mono=True)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
            sf.write(str(dst), audio, TARGET_SR, subtype='PCM_16')
            tmp.unlink(missing_ok=True)
        except:
            continue
    valid_ids.append(rec_id)
    txt_path = TRANS_DIR / f"{rec_id}.txt"
    if txt_path.exists():
        references[rec_id] = txt_path.read_text(encoding='utf-8')
    else:
        text = fetch_transcription(row['transcription_url_gcp'])
        if text and len(text) > 10:
            txt_path.write_text(text, encoding='utf-8')
            references[rec_id] = text

print(f"Valid: {len(valid_ids)} | References: {len(references)}")

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model = "openai/whisper-small",
    device = 0 if torch.cuda.is_available() else -1,
    chunk_length_s = 30,
    generate_kwargs = {"language": "hindi", "task": "transcribe"},
)

raw_asr_results = {}
for rec_id in tqdm(valid_ids, desc="Running ASR"):
    if rec_id not in references:
        continue
    out_path = RAW_ASR_DIR / f"{rec_id}.txt"
    if out_path.exists():
        raw_asr_results[rec_id] = out_path.read_text(encoding='utf-8')
        continue
    try:
        audio, sr = librosa.load(str(AUDIO_DIR / f"{rec_id}.wav"), sr=16000, mono=True)
        result = asr_pipe({"array": audio, "sampling_rate": sr})
        text = result["text"].strip()
        out_path.write_text(text, encoding='utf-8')
        raw_asr_results[rec_id] = text
    except Exception as e:
        print(f"ASR failed {rec_id}: {e}")

pairs_df = pd.DataFrame([
    {"id": rec_id, "raw_asr": raw_asr_results[rec_id], "reference": references[rec_id]}
    for rec_id in raw_asr_results if rec_id in references
])
pairs_df.to_csv(OUT_DIR / "asr_pairs.csv", index=False)
print(f"Pairs: {len(pairs_df)}")

wer_metric = evaluate.load("wer")

pairs_df['normalized_asr'] = pairs_df['raw_asr'].apply(normalize_numbers)
pairs_df['tagged_asr'] = pairs_df['raw_asr'].apply(lambda x: detect_english_words(x)[0])
pairs_df['english_words'] = pairs_df['raw_asr'].apply(lambda x: detect_english_words(x)[1])
pairs_df['en_word_count'] = pairs_df['english_words'].apply(len)
pairs_df['final_output'] = pairs_df['raw_asr'].apply(lambda x: full_pipeline(x)['tagged'])

wer_before = wer_metric.compute(predictions=pairs_df['raw_asr'].tolist(),         references=pairs_df['reference'].tolist())
wer_after  = wer_metric.compute(predictions=pairs_df['normalized_asr'].tolist(),  references=pairs_df['reference'].tolist())

print(f"WER before normalization: {wer_before*100:.2f}%")
print(f"WER after normalization: {wer_after*100:.2f}%")
print(f"Transcripts with English: {(pairs_df['en_word_count'] > 0).sum()}")
print(f"Total English words: {pairs_df['en_word_count'].sum()}")

pairs_df.to_csv(OUT_DIR / "pipeline_output.csv", index=False)
print(f"Output saved: {OUT_DIR / 'pipeline_output.csv'}")