import torch
import joblib
import numpy as np
import librosa
import pathlib
import os
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    data_dir = pathlib.Path("c:/Users/aniji/OneDrive/Desktop/IIITH speech analysis/data")
    files = list(data_dir.rglob("*.wav"))
    
    # Target layers to test: 6 (middle), 9 (late-middle)
    layers_data = {6: {"X": [], "y": [], "groups": []}, 
                   9: {"X": [], "y": [], "groups": []}}
    
    print(f"Extracting hidden states for {len(files)} files...")
    for f in tqdm(files[:400]): # Just doing a 400 sample balanced test for speed!
        emotion = f.parent.name.split('_')[-1].lower()
        speaker = f.parent.name.split('_')[0].upper()
        
        y, sr = librosa.load(f, sr=16000)
        inputs = extractor(y, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # outputs.hidden_states is a tuple of (embedding_layer, layer1, ..., layer12)
            
            for layer in [6, 9]:
                hidden = outputs.hidden_states[layer].squeeze(0).cpu().numpy()
                emb = np.mean(hidden, axis=0) # time mean pooling
                layers_data[layer]["X"].append(emb)
                layers_data[layer]["y"].append(emotion)
                layers_data[layer]["groups"].append(speaker)
                
    return layers_data

def evaluate(X, y, groups, layer_name):
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    accs = []
    print(f"\nEvaluating Layer {layer_name}")
    for tr_spk, te_spk in [("OAF", "YAF"), ("YAF", "OAF")]:
        tr_mask = (groups == tr_spk)
        te_mask = (groups == te_spk)
        if len(y[tr_mask]) == 0 or len(y[te_mask]) == 0:
            continue
            
        clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=2000, class_weight='balanced'))
        clf.fit(X[tr_mask], y[tr_mask])
        p = clf.predict(X[te_mask])
        a = accuracy_score(y[te_mask], p)
        accs.append(a)
        print(f"{tr_spk}->{te_spk}: {a:.4f}")
    
    if accs:
        print(f"Average: {np.mean(accs):.4f}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    layers_data = extract_features()
    for l in [6, 9]:
        evaluate(layers_data[l]["X"], layers_data[l]["y"], layers_data[l]["groups"], str(l))
