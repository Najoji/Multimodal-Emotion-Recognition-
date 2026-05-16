import torch
import numpy as np
import librosa
import pathlib
import random
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

MODEL_NAME = "facebook/wav2vec2-base"

def extract_features():
    print("Loading model for hidden states...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    device = "cpu"
    model = model.eval()
    data_dir = pathlib.Path("c:/Users/aniji/OneDrive/Desktop/IIITH speech analysis/data")
    files = list(data_dir.rglob("*.wav"))
    random.seed(42)
    random.shuffle(files)
    files = files[:600]
    layers_data = {6: {"X": [], "y": [], "groups": []}, 12: {"X": [], "y": [], "groups": []}}
    
    for f in tqdm(files):
        emotion = f.parent.name.split('_')[-1].lower()
        speaker = f.parent.name.split('_')[0].upper()
        y, _ = librosa.load(f, sr=16000)
        inputs = extractor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            for layer in [6, 12]:
                hidden = outputs.hidden_states[layer].squeeze(0).numpy()
                emb = np.mean(hidden, axis=0)
                layers_data[layer]["X"].append(emb)
                layers_data[layer]["y"].append(emotion)
                layers_data[layer]["groups"].append(speaker)
    return layers_data

def evaluate(X, y, groups, layer_name):
    X, y, groups = np.array(X), np.array(y), np.array(groups)
    print(f"\nEvaluating Layer {layer_name}")
    accs = []
    for tr_spk, te_spk in [("OAF", "YAF"), ("YAF", "OAF")]:
        tr_mask, te_mask = (groups == tr_spk), (groups == te_spk)
        if sum(tr_mask) == 0 or sum(te_mask) == 0: continue
        clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=2000))
        clf.fit(X[tr_mask], y[tr_mask])
        acc = accuracy_score(y[te_mask], clf.predict(X[te_mask]))
        accs.append(acc)
    if accs: print(f"Average for layer {layer_name}: {np.mean(accs):.4f}")

import warnings; warnings.filterwarnings('ignore')
layers_data = extract_features()
for l in [6, 12]: 
    evaluate(layers_data[l]["X"], layers_data[l]["y"], layers_data[l]["groups"], str(l))
