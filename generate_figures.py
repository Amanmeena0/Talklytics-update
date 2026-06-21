import os
import time
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder
import joblib

# Internal modules
from src.core import config
from src.ml.inference.fusion_inference import FusionModel
from src.features.acoustic.extractor import AcousticExtractor, AcousticFeatures
from src.features.linguistic.analyzer import LinguisticAnalyzer, LinguisticFeatures
from src.features.engagement.tracker import EngagementTracker
from src.ml.training.data_loader import DataLoader

# ────────────────────────────────────────────────────────────────────────── #
#  Styling & Config                                                          #
# ────────────────────────────────────────────────────────────────────────── #

OUTPUT_DIR = Path("report_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.autolayout"] = True

# Academic style: no heavy gridlines
sns.set_style("white")

# ────────────────────────────────────────────────────────────────────────── #
#  Helper: Dynamic Feature Parsing                                           #
# ────────────────────────────────────────────────────────────────────────── #

def get_feature_names():
    """Programmatically build the feature name list from extraction code."""
    acoustic_names = []
    
    # Acoustic Features
    # From modules.acoustic_extractor.AcousticFeatures
    for i in range(config.N_MFCC):
        acoustic_names.append(f"mfcc_mean_{i}")
    for i in range(config.N_MFCC):
        acoustic_names.append(f"mfcc_std_{i}")
    
    acoustic_names.extend(["pitch_mean", "pitch_std", "energy"])
    
    for i in range(7): # Spectral contrast mean length is 7
        acoustic_names.append(f"spectral_contrast_{i}")
        
    # Linguistic Features
    # From modules.linguistic_analyzer.LinguisticFeatures.to_vector()
    linguistic_names = [
        "sentiment_encoded",
        "sentiment_score",
        "buying_signal_count",
        "hesitation_count",
        "intent_count",
        "intent_confidence"
    ]
    
    full_names = acoustic_names + linguistic_names
    return full_names

# ────────────────────────────────────────────────────────────────────────── #
#  Data Loading                                                              #
# ────────────────────────────────────────────────────────────────────────── #

def load_data():
    """Load RAVDESS dataset using existing pipeline."""
    loader = DataLoader()
    ravdess_path = "archive"
    print(f"Loading data from {ravdess_path} ...")
    X, y = [], []
    wav_files = sorted(list(Path(ravdess_path).rglob("*.wav")))
    
    # Use 300 samples to keep runtime reasonable (approx 5-6 mins)
    wav_files = wav_files[:300]
    total = len(wav_files)
    print(f"Found {total} files. Starting extraction...")
    
    for i, path in enumerate(wav_files):
        if i % 20 == 0:
            print(f"  Processing {i}/{total} ...")
        parts = path.stem.split("-")
        if len(parts) < 3: continue
        from src.ml.training.data_loader import RAVDESS_TO_SCORE
        score = RAVDESS_TO_SCORE.get(parts[2], 3)
        
        audio, _ = librosa.load(str(path), sr=config.SAMPLE_RATE, mono=True)
        feats = loader._acoustic.extract(audio)
        from src.features.linguistic.analyzer import LinguisticFeatures
        ling = LinguisticFeatures()
        vec = np.concatenate([feats.to_vector(), ling.to_vector()])
        X.append(vec)
        y.append(score)
        
    return np.array(X), np.array(y)

# ────────────────────────────────────────────────────────────────────────── #
#  Main Logic                                                                #
# ────────────────────────────────────────────────────────────────────────── #

def main():
    feature_names = get_feature_names()
    print(f"Extracted {len(feature_names)} feature names.")
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model_wrapper = FusionModel()
    model_wrapper.load()
    
    if model_wrapper._clf is None:
        print("Model not found. Retraining...")
        model_wrapper.train(X_train, y_train)
        model_wrapper.save()
    
    clf = model_wrapper._clf
    le = model_wrapper._le
    
    # ────────────────────────────────────────────────────────────────────── #
    #  FIGURE 1 — Precision–Recall–F1 Bar Chart                              #
    # ────────────────────────────────────────────────────────────────────── #
    print("Generating Figure 1: Precision/Recall/F1 ...")
    y_pred_enc = clf.predict(X_test)
    report = classification_report(y_test, le.inverse_transform(y_pred_enc), output_dict=True)
    
    classes = [str(c) for c in le.classes_]
    metrics = ["precision", "recall", "f1-score"]
    
    df_metrics = []
    for cls in classes:
        for m in metrics:
            df_metrics.append({
                "Class": config.SCORE_LABELS.get(int(cls), cls),
                "Metric": m.capitalize(),
                "Value": report[cls][m]
            })
    df_fig1 = pd.DataFrame(df_metrics)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Class", y="Value", hue="Metric", data=df_fig1, palette="muted")
    min_val = df_fig1["Value"].min()
    plt.ylim(max(0, min_val - 0.05), 1.05)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=9, xytext=(0, 5), textcoords='offset points')
    plt.title("Per-Class Precision, Recall and F1-Score – ConvinceSense Model")
    plt.savefig(OUTPUT_DIR / "fig_precision_recall_f1.png")
    print("Figure 1 saved.")
    plt.close()

    # ────────────────────────────────────────────────────────────────────── #
    #  FIGURE 2 — Per-Class Accuracy Bar Chart                               #
    # ────────────────────────────────────────────────────────────────────── #
    print("Generating Figure 2: Per-Class Accuracy ...")
    cm = confusion_matrix(y_test, le.inverse_transform(y_pred_enc))
    per_class_acc = (cm.diagonal() / cm.sum(axis=1)) * 100
    class_names = [config.SCORE_LABELS.get(int(c), str(c)) for c in le.classes_]
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Blues_d", len(per_class_acc))
    sorted_indices = np.argsort(per_class_acc)
    color_map = [None] * len(per_class_acc)
    for i, idx in enumerate(sorted_indices): color_map[idx] = colors[i]
    ax = sns.barplot(x=per_class_acc, y=class_names, palette=color_map)
    plt.xlim(max(0, np.floor(per_class_acc.min() - 5)), 105)
    for i, p in enumerate(ax.patches):
        ax.text(p.get_width() - 5, p.get_y() + p.get_height()/2, f'{p.get_width():.2f}%', 
                va='center', ha='right', color='white', fontweight='bold')
    plt.title("Per-Class Classification Accuracy – ConvinceSense Model")
    plt.savefig(OUTPUT_DIR / "fig_per_class_accuracy.png")
    print("Figure 2 saved.")
    plt.close()

    # ────────────────────────────────────────────────────────────────────── #
    #  FIGURE 3 — Feature Importance Plot                                    #
    # ────────────────────────────────────────────────────────────────────── #
    print("Generating Figure 3: Feature Importance ...")
    importances = clf.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
    names, vals = zip(*feat_imp)
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x=list(vals), y=list(names), palette=sns.color_palette("Blues_r", len(vals)))
    for i, v in enumerate(vals): ax.text(v + 0.001, i, f'{v:.4f}', va='center')
    plt.title("Top 20 Feature Importances – Random Forest ConvinceSense Model")
    plt.savefig(OUTPUT_DIR / "fig_feature_importance.png")
    print("Figure 3 saved.")
    plt.close()

    # ────────────────────────────────────────────────────────────────────── #
    #  FIGURE 4 — n_estimators vs Accuracy Curve                             #
    # ────────────────────────────────────────────────────────────────────── #
    print("Generating Figure 4: n_estimators Curve ...")
    n_estimators_list = [5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200]
    train_accs, val_accs = [], []
    y_train_enc, y_test_enc = le.transform(y_train), le.transform(y_test)
    for n in n_estimators_list:
        temp_clf = RandomForestClassifier(n_estimators=n, random_state=42, class_weight="balanced")
        temp_clf.fit(X_train, y_train_enc)
        train_accs.append(accuracy_score(y_train_enc, temp_clf.predict(X_train)))
        val_accs.append(cross_val_score(temp_clf, X_train, y_train_enc, cv=5).mean())
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, train_accs, label="Train Accuracy", color="darkblue", marker='o')
    plt.plot(n_estimators_list, val_accs, label="Validation Accuracy (5-fold CV)", color="orange", linestyle='--', marker='s')
    plt.axvline(x=clf.n_estimators, color='grey', linestyle='--', label=f"Selected ({clf.n_estimators} trees)")
    plt.title("Effect of n_estimators on Model Accuracy – ConvinceSense")
    plt.legend(); plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.savefig(OUTPUT_DIR / "fig_n_estimators_accuracy.png")
    print("Figure 4 saved.")
    plt.close()

    # ────────────────────────────────────────────────────────────────────── #
    #  FIGURE 5 — Learning Curve                                             #
    # ────────────────────────────────────────────────────────────────────── #
    print("Generating Figure 5: Learning Curve ...")
    train_sizes, train_scores, test_scores = learning_curve(
        RandomForestClassifier(n_estimators=clf.n_estimators, random_state=42, class_weight="balanced"),
        X, le.transform(y), cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), color='darkblue', marker='o', label='Training Accuracy')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), color='orange', linestyle='--', marker='s', label='Cross-Validation Accuracy')
    plt.title("Learning Curve – ConvinceSense Random Forest Model")
    plt.legend(); plt.savefig(OUTPUT_DIR / "fig_learning_curve.png")
    print("Figure 5 saved.")
    plt.close()

    # ────────────────────────────────────────────────────────────────────── #
    #  FIGURE 6 — Processing Latency Chart                                   #
    # ────────────────────────────────────────────────────────────────────── #
    print("Generating Figure 6: Processing Latency ...")
    from src.pipelines.live_pipeline import ConvinceSensePipeline
    audio_files = list(Path("archive").rglob("*.wav"))[:30]
    stage_timings = collections.defaultdict(list)
    pipeline = ConvinceSensePipeline()
    for i, af in enumerate(audio_files):
        if i % 10 == 0: print(f"  Latency: {i}/{len(audio_files)} ...")
        audio, _ = librosa.load(str(af), sr=config.SAMPLE_RATE, mono=True)
        t0 = time.perf_counter(); clean = pipeline.preprocessor.process(audio); stage_timings["AudioPreprocessor"].append(time.perf_counter() - t0)
        if clean is None: continue
        t0 = time.perf_counter(); acoustic = pipeline.acoustic.extract(clean); stage_timings["AcousticExtractor"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); transcript = pipeline.asr.transcribe(clean); stage_timings["SpeechRecognizer"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); linguistic = pipeline.nlp.analyze(transcript); stage_timings["LinguisticAnalyzer"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); pipeline.model.predict(acoustic, linguistic); stage_timings["FusionModel"].append(time.perf_counter() - t0)
    stages = list(stage_timings.keys())
    means = [np.mean(stage_timings[s]) * 1000 for s in stages]
    stds = [np.std(stage_timings[s]) * 1000 for s in stages]
    plt.figure(figsize=(12, 7))
    colors = ["green" if m < 100 else "orange" if m < 200 else "red" for m in means]
    ax = sns.barplot(x=stages, y=means, palette=colors)
    plt.errorbar(x=stages, y=means, yerr=stds, fmt='none', c='black', capsize=5)
    plt.axhline(y=1500, color='red', linestyle='--', label="Latency Budget (1500ms)")
    plt.axhline(y=sum(means), color='green', linestyle='--', label=f"Total Mean: {sum(means):.2f}ms")
    plt.title("Per-Module Processing Latency – ConvinceSense Pipeline")
    plt.savefig(OUTPUT_DIR / "fig_processing_latency.png")
    print("Figure 6 saved.")
    plt.close()

    # ────────────────────────────────────────────────────────────────────── #
    #  FIGURE 7 — Score Distribution Histogram                               #
    # ────────────────────────────────────────────────────────────────────── #
    print("Generating Figure 7: Score Distribution ...")
    counts = collections.Counter(le.inverse_transform(y_pred_enc))
    all_classes = sorted(le.classes_)
    class_counts = [counts.get(c, 0) for c in all_classes]
    class_names = [config.SCORE_LABELS.get(int(c), str(c)) for c in all_classes]
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=class_names, y=class_counts, palette=plt.cm.RdYlGn(np.linspace(0, 1, len(all_classes))))
    plt.title("Convincingness Score Distribution – ConvinceSense Test Set")
    plt.savefig(OUTPUT_DIR / "fig_score_distribution.png")
    print("Figure 7 saved.")
    plt.close()

    # ────────────────────────────────────────────────────────────────────── #
    #  FIGURE 8 — ROC Curve (One-vs-Rest)                                     #
    # ────────────────────────────────────────────────────────────────────── #
    print("Generating Figure 8: ROC Curves ...")
    y_test_bin = label_binarize(y_test, classes=le.classes_)
    y_score = clf.predict_proba(X_test)
    plt.figure(figsize=(10, 8))
    for i in range(len(le.classes_)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'Class {config.SCORE_LABELS.get(int(le.classes_[i]), le.classes_[i])} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves (One-vs-Rest) – ConvinceSense Model')
    plt.legend(); plt.savefig(OUTPUT_DIR / "fig_roc_curve.png")
    print("Figure 8 saved.")
    plt.close()

    # ────────────────────────────────────────────────────────────────────── #
    #  FIGURE 9 — Annotated Engagement Trajectory                            #
    # ────────────────────────────────────────────────────────────────────── #
    print("Generating Figure 9: Annotated Trajectory ...")
    tracker = EngagementTracker()
    alpha = config.INTENT_CONFIDENCE_SMOOTHING
    smoothed_scores, raw_scores, last_smoothed = [], [], None
    test_wavs = list(Path("archive").rglob("*.wav"))[:20]
    pipeline = ConvinceSensePipeline()
    for i, wav in enumerate(test_wavs):
        if i % 5 == 0: print(f"  Trajectory: {i}/{len(test_wavs)} ...")
        audio, _ = librosa.load(str(wav), sr=config.SAMPLE_RATE, mono=True)
        clean = pipeline.preprocessor.process(audio)
        if clean is None: continue
        acoustic = pipeline.acoustic.extract(clean)
        transcript = pipeline.asr.transcribe(clean)
        linguistic = pipeline.nlp.analyze(transcript)
        score, _ = pipeline.model.predict(acoustic, linguistic)
        tracker.add(score=score, transcript=transcript, sentiment=linguistic.sentiment_label,
                    buying_signals=linguistic.buying_signals, hesitations=linguistic.hesitations,
                    detected_intents=linguistic.detected_intents, intent_confidence=linguistic.intent_confidence)
        if last_smoothed is None: last_smoothed = float(score)
        else: last_smoothed = alpha * last_smoothed + (1 - alpha) * score
        smoothed_scores.append(last_smoothed); raw_scores.append(score)
    plt.figure(figsize=(12, 6))
    plt.plot(tracker.timestamps, raw_scores, 'o', color='lightgrey', alpha=0.5, label='Raw Score')
    plt.plot(tracker.timestamps, smoothed_scores, '-', color='blue', linewidth=2, label=f'EMA (α={alpha})')
    plt.axhspan(1, 2.5, color='red', alpha=0.1); plt.axhspan(3.5, 5, color='green', alpha=0.1)
    plt.yticks([1, 2, 3, 4, 5], [config.SCORE_LABELS[i] for i in [1, 2, 3, 4, 5]])
    plt.title("Annotated Engagement Score Trajectory – ConvinceSense Session")
    plt.savefig(OUTPUT_DIR / "fig_engagement_annotated.png")
    print("Figure 9 saved.")
    plt.close()

    print("\n" + "="*80)
    print(f"{'Filename':<30} | {'Figure Title':<45} | {'Size (KB)':<10}")
    print("-" * 80)
    for fig in sorted(OUTPUT_DIR.glob("*.png")):
        size_kb = fig.stat().st_size / 1024
        print(f"{fig.name:<30} | {'Computed from data':<45} | {size_kb:>8.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
