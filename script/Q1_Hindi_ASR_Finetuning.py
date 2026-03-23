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
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Any
from datasets import Dataset, Audio as HFAudio, load_dataset as hf_load
from transformers import (
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    pipeline,
)
import evaluate

BASE_DIR = Path("/teamspace/studios/this_studio")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" / "q1"
AUDIO_DIR = OUT_DIR / "audio_16k"
CHUNKS_DIR = OUT_DIR / "chunks"
TRANS_DIR = OUT_DIR / "transcriptions"
MODEL_DIR = OUT_DIR / "model"

for d in [OUT_DIR, AUDIO_DIR, CHUNKS_DIR, TRANS_DIR, MODEL_DIR]:
    d.mkdir(exist_ok=True, parents=True)

CSV_PATH = DATA_DIR / "FT Data - data.csv"
MODEL_NAME = "openai/whisper-small"
TARGET_SR = 16000
MAX_CHUNK_DURATION = 25.0
MAX_LABEL_LENGTH = 448


def fix_gcp_url(url):
    match = re.search(r'hq_data/hi/(\d+)/(\d+)_(.*)', url)
    if match:
        folder_id, recording_id, suffix = match.groups()
        return f'https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_{suffix}'
    return url


def normalize_hindi_text(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\u0900-\u097F\s0-9]', '', text)
    text = re.sub(r'\b[0-9]+\b', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def fetch_json(url, timeout=30):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except:
        return None


def fetch_transcription(url):
    data = fetch_json(url)
    if not data:
        return None
    if isinstance(data, list):
        return " ".join([s["text"].strip() for s in data if s.get("text", "").strip()])
    if isinstance(data, dict):
        return data.get("text") or data.get("transcription", "")
    return None


def download_and_resample(url, dst_path):
    tmp = dst_path.with_suffix(".tmp.wav")
    try:
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        audio, _ = librosa.load(str(tmp), sr=TARGET_SR, mono=True)
        if len(audio) < TARGET_SR:
            tmp.unlink(missing_ok=True)
            return False
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        sf.write(str(dst_path), audio, TARGET_SR, subtype='PCM_16')
        tmp.unlink(missing_ok=True)
        return True
    except:
        tmp.unlink(missing_ok=True)
        return False


def save_chunk(rec_id, chunk_idx, segs, audio, sr):
    chunk = audio[int(segs[0]['start'] * sr):min(int(segs[-1]['end'] * sr), len(audio))]
    if len(chunk) < sr:
        return None
    text = normalize_hindi_text(" ".join([s['text'].strip() for s in segs if s.get('text', '').strip()]))
    if not text:
        return None
    chunk_id   = f"{rec_id}_chunk{chunk_idx:03d}"
    chunk_path = CHUNKS_DIR / f"{chunk_id}.wav"
    sf.write(str(chunk_path), chunk, sr, subtype='PCM_16')
    return {"id": chunk_id, "audio_path": str(chunk_path), "text": text, "duration": len(chunk) / sr}


def prepare_sample(batch, feature_extractor, tokenizer):
    audio, _ = librosa.load(batch["audio_path"], sr=16000, mono=True)
    batch["input_features"] = feature_extractor(audio, sampling_rate=16000).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"], max_length=MAX_LABEL_LENGTH, truncation=True).input_ids
    return batch


@dataclass
class DataCollator:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


df = pd.read_csv(CSV_PATH)
df['transcription_url_gcp'] = df['transcription_url_gcp'].apply(fix_gcp_url)
df['rec_url_gcp'] = df['rec_url_gcp'].apply(fix_gcp_url)
print(f"Recordings: {len(df)} | Duration: {df['duration'].sum()/3600:.2f}h")

transcriptions = {}
for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcriptions"):
    rec_id    = str(row['recording_id'])
    save_path = TRANS_DIR / f"{rec_id}.txt"
    if save_path.exists():
        transcriptions[rec_id] = save_path.read_text(encoding='utf-8')
        continue
    text = fetch_transcription(row['transcription_url_gcp'])
    if text:
        text = normalize_hindi_text(text)
        if len(text) > 20:
            save_path.write_text(text, encoding='utf-8')
            transcriptions[rec_id] = text

audio_ok = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Audio"):
    rec_id = str(row['recording_id'])
    if rec_id not in transcriptions:
        continue
    dst = AUDIO_DIR / f"{rec_id}.wav"
    if dst.exists() or download_and_resample(row['rec_url_gcp'], dst):
        audio_ok.append(rec_id)

chunk_manifest = []
for rec_id in tqdm(audio_ok, desc="Segmenting"):
    url  = df[df['recording_id'] == int(rec_id)]['transcription_url_gcp'].values[0]
    segs = fetch_json(url)
    if not segs or not isinstance(segs, list):
        continue
    audio, sr = librosa.load(str(AUDIO_DIR / f"{rec_id}.wav"), sr=TARGET_SR, mono=True)
    current_segs, current_dur, chunk_idx = [], 0.0, 0
    for seg in segs:
        seg_dur = seg['end'] - seg['start']
        if current_dur + seg_dur > MAX_CHUNK_DURATION and current_segs:
            result = save_chunk(rec_id, chunk_idx, current_segs, audio, sr)
            if result:
                chunk_manifest.append(result)
            chunk_idx += 1
            current_segs, current_dur = [], 0.0
        current_segs.append(seg)
        current_dur += seg_dur
    if current_segs:
        result = save_chunk(rec_id, chunk_idx, current_segs, audio, sr)
        if result:
            chunk_manifest.append(result)

manifest_df = pd.DataFrame(chunk_manifest)
print(f"Chunks: {len(manifest_df)} | Duration: {manifest_df['duration'].sum()/3600:.2f}h")

train_df, val_df = train_test_split(manifest_df, test_size=0.1, random_state=42)

processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="Hindi", task="transcribe")
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer

train_ds = Dataset.from_dict({"audio_path": train_df["audio_path"].tolist(), "sentence": train_df["text"].tolist()})
val_ds = Dataset.from_dict({"audio_path": val_df["audio_path"].tolist(),   "sentence": val_df["text"].tolist()})

train_ds = train_ds.map(lambda b: prepare_sample(b, feature_extractor, tokenizer), remove_columns=train_ds.column_names, num_proc=1)
val_ds = val_ds.map(lambda b: prepare_sample(b, feature_extractor, tokenizer),   remove_columns=val_ds.column_names,   num_proc=1)
train_ds = train_ds.filter(lambda x: len(x["labels"]) <= MAX_LABEL_LENGTH)
val_ds = val_ds.filter(lambda x: len(x["labels"]) <= MAX_LABEL_LENGTH)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

wer_metric = evaluate.load("wer")
data_collator = DataCollator(processor=processor)


def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4)}


model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.generation_config.language = "hindi"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.dropout = 0.0
model.config.attention_dropout = 0.0

training_args = Seq2SeqTrainingArguments(
    output_dir = str(OUT_DIR / "checkpoints"),
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 8,
    gradient_accumulation_steps = 2,
    learning_rate = 1e-5,
    warmup_steps = 200,
    num_train_epochs = 10,
    lr_scheduler_type = "cosine",
    gradient_checkpointing = True,
    fp16 = True,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    save_total_limit = 3,
    logging_steps = 25,
    load_best_model_at_end = True,
    metric_for_best_model = "wer",
    greater_is_better = False,
    predict_with_generate = True,
    generation_max_length = 225,
    dataloader_num_workers = 2,
    report_to = ["tensorboard"],
    push_to_hub = False,
)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
    processing_class = feature_extractor,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
trainer.save_model(str(MODEL_DIR))
processor.save_pretrained(str(MODEL_DIR))
print(f"Model saved: {MODEL_DIR}")

fleurs_test = hf_load("google/fleurs", "hi_in", split="test", trust_remote_code=True)
fleurs_test = fleurs_test.cast_column("audio", HFAudio(sampling_rate=16000))
baseline_pipe = pipeline("automatic-speech-recognition", model=MODEL_NAME,     device=0, chunk_length_s=30, generate_kwargs={"language": "hindi", "task": "transcribe"})
finetuned_pipe = pipeline("automatic-speech-recognition", model=str(MODEL_DIR), device=0, chunk_length_s=30, generate_kwargs={"language": "hindi", "task": "transcribe"})

baseline_preds,  baseline_refs  = [], []
finetuned_preds, finetuned_refs = [], []

for sample in tqdm(fleurs_test, desc="Baseline inference"):
    baseline_preds.append(baseline_pipe(sample["audio"]["array"])["text"])
    baseline_refs.append(sample["transcription"])

for sample in tqdm(fleurs_test, desc="Fine-tuned inference"):
    finetuned_preds.append(finetuned_pipe(sample["audio"]["array"])["text"])
    finetuned_refs.append(sample["transcription"])

baseline_wer  = wer_metric.compute(predictions=baseline_preds,  references=baseline_refs)
finetuned_wer = wer_metric.compute(predictions=finetuned_preds, references=finetuned_refs)

pd.DataFrame({
    "model": ["Whisper-small Baseline", "Whisper-small Fine-tuned"],
    "test_set": ["FLEURS Hindi (hi_in)", "FLEURS Hindi (hi_in)"],
    "wer": [f"{baseline_wer*100:.2f}%", f"{finetuned_wer*100:.2f}%"],
}).to_csv(OUT_DIR / "wer_results.csv", index=False)

print(f"Baseline WER:   {baseline_wer*100:.2f}%")
print(f"Fine-tuned WER: {finetuned_wer*100:.2f}%")

per_utt  = [{"reference": ref, "hypothesis": pred, "wer": wer_metric.compute(predictions=[pred], references=[ref])} for pred, ref in zip(finetuned_preds, finetuned_refs)]
error_df = pd.DataFrame(per_utt)
error_df = error_df[error_df["wer"] > 0].reset_index(drop=True)

low  = error_df[error_df["wer"] <= 0.3]
mid  = error_df[(error_df["wer"] > 0.3) & (error_df["wer"] <= 0.7)]
high = error_df[error_df["wer"] > 0.7]

n_low, n_mid, n_high = min(9, len(low)), min(9, len(mid)), min(9, len(high))
while n_low + n_mid + n_high < 25:
    if n_high < len(high): n_high += 1
    elif n_mid < len(mid): n_mid += 1
    elif n_low < len(low): n_low += 1
    else: break

pd.concat([low.sample(n_low, random_state=42), mid.sample(n_mid, random_state=42), high.sample(n_high, random_state=42)]).reset_index(drop=True).to_csv(OUT_DIR / "sampled_errors.csv", index=False)

pipe_before = pipeline("automatic-speech-recognition", model=str(MODEL_DIR), device=0, chunk_length_s=30, generate_kwargs={"language": "hindi", "task": "transcribe"})
pipe_after  = pipeline("automatic-speech-recognition", model=str(MODEL_DIR), device=0, chunk_length_s=30, generate_kwargs={"language": "hindi", "task": "transcribe", "repetition_penalty": 1.3, "no_repeat_ngram_size": 3})

target_samples = [s for s in fleurs_test if s["transcription"] in error_df[error_df["wer"] > 0.8]["reference"].tolist()][:6]
before_preds, after_preds, refs = [], [], []
for sample in target_samples:
    before_preds.append(pipe_before(sample["audio"]["array"])["text"])
    after_preds.append(pipe_after(sample["audio"]["array"])["text"])
    refs.append(sample["transcription"])

print(f"Repetition Penalty Fix — Before: {wer_metric.compute(predictions=before_preds, references=refs)*100:.2f}% | After: {wer_metric.compute(predictions=after_preds, references=refs)*100:.2f}%")