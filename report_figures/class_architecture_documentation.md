# ConvinceSense — Class Architecture Documentation

This document provides a class-by-class analysis of the ConvinceSense system architecture, detailing the purpose, state, methods, relationships, and data flows for every class in the codebase.

---

## 1. Core Orchestration & Hardware Layer

### Class: `AudioCapture`
* **File**: [`src/core/audio_capture.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/core/audio_capture.py)
* **Purpose**: Manages low-level PortAudio input streams, buffers incoming PCM audio blocks, handles overflow warnings, and provides backlog-pruning to ensure real-time synchronization.

#### Variables
| Name | Type | Purpose | Initialized / Updated |
|------|------|---------|-----------------------|
| `self.sample_rate` | `int` | Audio sample rate (e.g., 16000 Hz) | `__init__` (constant) |
| `self.channels` | `int` | Number of channels (e.g., 1 for mono) | `__init__` (constant) |
| `self.segment_samples` | `int` | Total samples in one analysis segment | `__init__` (constant) |
| `self.device` | `int \| None` | PortAudio device index override | `__init__` (constant) |
| `self._blocksize` | `int` | Sample size per callback invocation (512) | `__init__` (constant) |
| `self._block_queue` | `queue.Queue` | FIFO queue for incoming raw blocks | `__init__` / Pushed in `_callback`, popped in `get_segment` |
| `self._buffer` | `list[np.ndarray]`| Accumulator for current segment assembly | `__init__` / Cleared/Appended in `get_segment` |
| `self._stream` | `sd.InputStream` | PortAudio audio stream handler | `start` / Stopped and closed in `stop` |
| `self._running` | `bool` | Dictates if recording stream is active | `__init__` (False) / `True` in `start`, `False` in `stop` |

#### Methods
- **`__init__(self, sample_rate, channels, segment_duration, device)`**: Sets stream parameters and detects input device if `device` is `None`.
- **`start(self)`**: Sets `_running=True`, drains any leftover blocks in `_block_queue`, opens and starts the `sd.InputStream`.
- **`stop(self)`**: Sets `_running=False`, stops and closes `self._stream`, and sets it to `None`.
- **`get_segment(self, timeout=10.0) -> np.ndarray | None`**:
  - Drains/discards old blocks via `_prune_backlog()`.
  - Blocks and reads incoming raw blocks from `_block_queue` until `segment_samples` are accumulated.
  - Returns the concatenated `float32` NumPy segment array, keeping any remainder for the next iteration.
  - *Called by*: `ConvinceSensePipeline._run()`.
- **`active_device_name(self) -> str` (Property)**: Returns the active device's friendly query name or fallback index string.
  - *Called by*: Dashboard UI controls in `app.py`.
- **`_prune_backlog(self)`**: Checks if `_block_queue.qsize() > 140` (1.5 segments). If so, drains the oldest blocks and clears `self._buffer` to keep transcription synchronized with real-time.
  - *Called by*: `get_segment()`.
- **`_callback(self, indata, frames, time_info, status)`**: Sounddevice input stream handler. Runs on a real-time OS thread; performs a lock-free `put_nowait` of `indata` into `_block_queue`.
  - *Called by*: PortAudio wrapper callback system.

#### Relationships
- **Composition**: `ConvinceSensePipeline` instantiates `AudioCapture`.
- **Dependencies**: `sounddevice`, `numpy`, `queue`, `warnings`.

#### Data Flow
```
[Microphone] ──> _callback() ──> _block_queue ──> get_segment() (Concatenates) ──> float32 segment
```

---

### Class: `ConvinceSensePipeline`
* **File**: [`src/pipelines/live_pipeline.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/pipelines/live_pipeline.py)
* **Purpose**: Facade pattern orchestrator that initializes all feature and ML classes, runs the background thread processing loop, and publishes results to `output_q`.

#### Variables
| Name | Type | Purpose | Initialized / Updated |
|------|------|---------|-----------------------|
| `self.capture` | `AudioCapture` | Audio recording handler | `__init__` |
| `self.preprocessor` | `AudioPreprocessor` | Silence filter + normalization | `__init__` |
| `self.acoustic` | `AcousticExtractor` | Prosodic feature extractor | `__init__` |
| `self.diarizer` | `SpeakerDiarizer` | Conversational speaker separator | `__init__` |
| `self.asr` | `SpeechRecognizer` | Speech-to-text transcriber | `__init__` |
| `self.nlp` | `LinguisticAnalyzer` | Sentiment and intent classifier | `__init__` |
| `self.model` | `FusionModel` | Scikit-learn RF predictor | `__init__` |
| `self.tracker` | `EngagementTracker` | Session timeline registry | `__init__` |
| `self.summarizer` | `LLMSummarizer` | Gemini summaries creator | `__init__` |
| `self.output_q` | `queue.Queue` | Records queue published to dashboard | `__init__` |
| `self._thread` | `threading.Thread` | Pipeline loop background runner | `None` / Spawned in `start()`, joined in `stop()` |
| `self._running` | `bool` | Background thread loop control flag | `False` / `True` in `start()`, `False` in `stop()` |

#### Methods
- **`__init__(self)`**: Instantiates all subsystem classes and loads `self.model`.
  - *Called by*: `app.py` or `main.py`.
- **`get_summary(self) -> str`**: Returns AI-generated summary report text from the summarizer.
  - *Called by*: Dashboard UI summary button.
- **`start(self)`**: Spawns and starts the daemon processing thread (`self._run()`).
  - *Called by*: `app.py` or `main.py`.
- **`stop(self)`**: Flags `_running=False`, stops `self.capture`, and joins the processing thread.
  - *Called by*: `app.py` or `main.py`.
- **`_run(self)`**: Loops continuously while `_running=True`:
  1. Grabs segments from `self.capture`.
  2. Runs `self.preprocessor` (filters silence).
  3. Extracts features via `self.acoustic` and transcribes via `self.asr` in sequence.
  4. Runs linguistic analytics on the transcript via `self.nlp`.
  5. Computes score via `self.model` and speaker via `self.diarizer`.
  6. Adds metadata to `self.tracker`, yielding an `EngagementRecord` pushed to `output_q`.
  - *Called by*: `self._thread`.

#### Relationships
- **Composition**: Consists of all subsystems (`AudioCapture`, `AcousticExtractor`, etc.).
- **Dependencies**: `threading`, `queue`, `numpy`.

---

## 2. Acoustic Feature Processing Layer

### Class: `AudioPreprocessor`
* **File**: [`src/features/acoustic/preprocessor.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/features/acoustic/preprocessor.py)
* **Purpose**: Normalizes segment amplitudes and skips processing on segments below the RMS silence threshold.

#### Variables
| Name | Type | Purpose | Initialized / Updated |
|------|------|---------|-----------------------|
| `self.silence_threshold` | `float` | RMS power value baseline | `__init__` (constant) |

#### Methods
- **`__init__(self, silence_threshold=0.001)`**: Sets the RMS silence gate threshold.
- **`process(self, segment: np.ndarray) -> np.ndarray | None`**: Checks RMS power of the float32 array. Returns `None` if it is silent, otherwise returns peak-normalized array (`segment / np.max(np.abs(segment))`).
  - *Called by*: `ConvinceSensePipeline._run()`.

#### Relationships
- **Composition**: Instantiated inside `ConvinceSensePipeline`.

---

### Class: `AcousticFeatures`
* **File**: [`src/features/acoustic/extractor.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/features/acoustic/extractor.py)
* **Purpose**: Dataclass holding all extracted acoustic/prosodic properties of a segment.

#### Variables
| Name | Type | Purpose | Initialized / Updated |
|------|------|---------|-----------------------|
| `mfcc_mean` | `np.ndarray` | 13 average MFCC coefficients | Dataclass field |
| `mfcc_std` | `np.ndarray` | 13 standard deviations of MFCCs | Dataclass field |
| `pitch_mean` | `float` | Average fundamental frequency (F0) | Dataclass field |
| `pitch_std` | `float` | Standard deviation of F0 | Dataclass field |
| `energy` | `float` | Average root-mean-square energy | Dataclass field |
| `spectral_contrast_mean` | `np.ndarray` | 7-band spectral contrast averages | Dataclass field |

#### Methods
- **`to_vector(self) -> np.ndarray`**: Concatenates features into a single flat `float32` array of length 36.
  - *Called by*: `SpeakerDiarizer.identify_speaker()`, `FusionModel._build_vector()`.

---

### Class: `AcousticExtractor`
* **File**: [`src/features/acoustic/extractor.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/features/acoustic/extractor.py)
* **Purpose**: Extracts prosodic properties from raw audio segments using the Librosa library.

#### Variables
| Name | Type | Purpose | Initialized / Updated |
|------|------|---------|-----------------------|
| `self.sample_rate` | `int` | Sample rate used for feature scaling | `__init__` (constant) |
| `self.n_mfcc` | `int` | Quantization degree of Mel-frequency bands | `__init__` (constant) |

#### Methods
- **`extract(self, segment: np.ndarray) -> AcousticFeatures`**: Calculates librosa MFCCs, fundamental frequency via `pyin` (autocorrelation method), RMS energy, and spectral contrast.
  - *Called by*: `ConvinceSensePipeline._run()`.

#### Relationships
- **Dependencies**: `librosa`, `numpy`.

---

### Class: `SpeakerDiarizer`
* **File**: [`src/features/acoustic/diarizer.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/features/acoustic/diarizer.py)
* **Purpose**: Groups audio segments into cluster representations of Speaker 1 or Speaker 2 using online clustering.

#### Variables
| Name | Type | Purpose | Initialized / Updated |
|------|------|---------|-----------------------|
| `self.n_speakers` | `int` | Number of speaker categories (2) | `__init__` (constant) |
| `self.kmeans` | `MiniBatchKMeans` | Online clustering model | `__init__` / Modified in `identify_speaker` |
| `self._is_initialized` | `bool` | Centroid state tracking flag | `__init__` (False) / `True` in `identify_speaker` |

#### Methods
- **`identify_speaker(self, feature_vector: np.ndarray) -> str`**: Performs online partial fits on acoustic vectors. Categorizes segments as `"Speaker 1"` or `"Speaker 2"`.
  - *Called by*: `ConvinceSensePipeline._run()`.

#### Relationships
- **Dependencies**: `scikit-learn` (`MiniBatchKMeans`), `numpy`.

---

## 3. Linguistic & NLP Layer

### Class: `SpeechRecognizer`
* **File**: [`src/features/linguistic/recognizer.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/features/linguistic/recognizer.py)
* **Purpose**: Wraps Faster-Whisper to transcribe raw speech arrays to text.

#### Variables
| Name | Type | Purpose | Initialized / Updated |
|------|------|---------|-----------------------|
| `self.language` | `str` | Language targeted (default: "en") | `__init__` (constant) |
| `self._model` | `WhisperModel` | Internal Whisper runner | `__init__` (model load) |

#### Methods
- **`transcribe(self, segment: np.ndarray, sample_rate) -> str`**: Runs greedy transcription (`beam_size=1`) with a Silero VAD filter and `condition_on_previous_text=False` to prevent duplicate loops.
  - *Called by*: `ConvinceSensePipeline._run()`.

#### Relationships
- **Dependencies**: `faster-whisper`.

---

### Class: `LinguisticFeatures`
* **File**: [`src/features/linguistic/analyzer.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/features/linguistic/analyzer.py)
* **Purpose**: Dataclass storing NLP analysis outputs.

#### Variables
| Name | Type | Purpose | Initialized / Updated |
|------|------|---------|-----------------------|
| `sentiment_label` | `str` | Sentiment label ("POSITIVE", "NEGATIVE", "NEUTRAL") | Dataclass field |
| `sentiment_score` | `float` | Model confidence value | Dataclass field |
| `buying_signals` | `list[str]` | Detected buying signals | Dataclass field |
| `hesitations` | `list[str]` | Detected hesitations | Dataclass field |
| `detected_intents` | `list[str]` | Intent categories detected | Dataclass field |
| `intent_confidence` | `float` | Intent confidence density score | Dataclass field |

#### Methods
- **`to_vector(self) -> np.ndarray`**: Encodes text features to a flat numeric vector of length 6.
  - *Called by*: `FusionModel._build_vector()`.

---

### Class: `LinguisticAnalyzer`
* **File**: [`src/features/linguistic/analyzer.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/features/linguistic/analyzer.py)
* **Purpose**: Runs sentiment analysis using DistilBERT and performs rule-based intent matching on transcripts.

#### Variables
| Name | Type | Purpose | Initialized / Updated |
|------|------|---------|-----------------------|
| `self._sentiment` | `pipeline` | HuggingFace pipeline for sentiment | `__init__` (model load) |

#### Methods
- **`analyze(self, text: str) -> LinguisticFeatures`**: Runs sentiment classification, filters low-confidence negative labels to neutral, checks for keyword matches, and evaluates intent patterns.
  - *Called by*: `ConvinceSensePipeline._run()`.
- **`_detect_intents(text_lower: str) -> tuple[list[str], float]` (Static)**: Uses length-dampened density formulas to score matching regex categories.
  - *Called by*: `analyze()`.

#### Relationships
- **Dependencies**: `transformers` (HuggingFace pipeline), `numpy`.

---

## 4. Machine Learning & Post-Call Summary Layer

### Class: `FusionModel`
* **File**: [`src/ml/inference/fusion_inference.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/ml/inference/fusion_inference.py)
* **Purpose**: Concatenates features into a 42-dimensional vector and runs it through a Random Forest model to calculate an engagement score.

#### Variables
| Name | Type | Purpose | Initialized / Updated |
|------|------|---------|-----------------------|
| `self._clf` | `RandomForestClassifier` | Serialized RF classifier | Loaded in `load()` |
| `self._le` | `LabelEncoder` | Serialized label classes encoder | Loaded in `load()` |

#### Methods
- **`predict(self, acoustic, linguistic) -> tuple[int, float]`**: Combines features. Runs `predict_raw()`, or falls back to rule-based `_heuristic()` if the model is not found on disk.
  - *Called by*: `ConvinceSensePipeline._run()`.
- **`predict_raw(self, feature_vec: np.ndarray) -> tuple[int, float]`**: Runs model prediction and returns the score and probability confidence.
  - *Called by*: `predict()`.
- **`load(self, model_path, encoder_path)`**: Loads pickled model files from the file system.
  - *Called by*: `ConvinceSensePipeline.__init__()`.
- **`_build_vector(acoustic, linguistic) -> np.ndarray` (Static)**: Combines acoustic (36) and linguistic (6) features into a 42-D array.
- **`_heuristic(acoustic, linguistic) -> tuple[int, float]` (Static)**: Fallback rule system calculating engagement score based on linguistic rules.

#### Relationships
- **Dependencies**: `scikit-learn`, `joblib`, `numpy`.

---

### Class: `LLMSummarizer`
* **File**: [`src/features/summarization/llm_summarizer.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/features/summarization/llm_summarizer.py)
* **Purpose**: Leverages the Gemini API to construct post-call summaries, BANT assessments, and follow-up email drafts.

#### Variables
| Name | Type | Purpose | Initialized / Updated |
|------|------|---------|-----------------------|
| `self.api_key` | `str \| None` | Gemini credentials string | `__init__` |
| `self.client` | `genai.Client \| None` | Gemini API client | `__init__` |

#### Methods
- **`generate_summary(self, records: list[EngagementRecord]) -> str`**: Concatenates transcripts chronologically and generates the summary using Gemini, or prints a raw text fallback if no API key is available.
  - *Called by*: `ConvinceSensePipeline.get_summary()`.

#### Relationships
- **Dependencies**: `google-genai`.

---

## 5. Metrics & State Tracking Layer

### Class: `EngagementRecord`
* **File**: [`src/features/engagement/tracker.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/features/engagement/tracker.py)
* **Purpose**: Dataclass container holding all metrics and results for a single conversation segment.

#### Variables
All class fields (`timestamp`, `score`, `transcript`, `sentiment`, `buying_signals`, `hesitations`, `detected_intents`, `intent_confidence`, `recommendation`, `energy`, `confidence`, `speaker`) represent a single captured segment record.

---

### Class: `EngagementTracker`
* **File**: [`src/features/engagement/tracker.py`](file:///Users/amanmeena/Documents/Work/ConvinceSense-Update/src/features/engagement/tracker.py)
* **Purpose**: Stores historical records of a live call session, tracks cumulative averages, and smooths intent metrics.

#### Variables
| Name | Type | Purpose | Initialized / Updated |
|------|------|---------|-----------------------|
| `self._records` | `list[EngagementRecord]` | History list of the session | `__init__` / Modified in `add()`, `reset()` |
| `self._start` | `float` | Start timestamp of the session | `__init__` / Modified in `reset()` |

#### Methods
- **`add(self, score, transcript, sentiment, buying_signals, hesitations, ...) -> EngagementRecord`**: Smooths the intent confidence using an exponential moving average (EMA) (`α = 0.7`), creates an `EngagementRecord`, and appends it to `self._records`.
- **`records(self) -> list[EngagementRecord]` (Property)**: Returns the current list of session records.
- **`timestamps(self) -> list[float]` (Property)**: Returns a list of timestamps for the timeline chart.
- **`scores(self) -> list[int]` (Property)**: Returns a list of scores for the timeline chart.
- **`average_score(self) -> float` (Property)**: Calculates the average score across all records.
- **`latest(self) -> EngagementRecord | None` (Property)**: Returns the last record.
- **`reset(self)`**: Clears the records list and resets `self._start` to the current system time.

---

## 6. End-to-End System Data Flow

```
[System Audio Input] 
       │ (PortAudio Callback Thread)
       ▼
   [AudioCapture] (Accumulates block inputs)
       │ (Prunes backlog)
       ▼
   [AudioPreprocessor] (Filters silence, normalizes peak amplitude)
       │ (Processed Float32 Array Segment)
       ▼
 ┌─────┴──────────────────────────────────────────┐
 │                                                │
 ▼ (Librosa)                                      ▼ (Faster-Whisper)
[AcousticExtractor] (MFCCs, Pitch, Energy)       [SpeechRecognizer] (ASR Transcription)
 │                                                │
 │ (36-D Vector)                                  ▼ (DistilBERT / Patterns)
 │                                               [LinguisticAnalyzer] (Sentiment, Intents)
 │                                                │
 └─────┬──────────────────────────────────────────┘ (6-D Vector)
       ▼
   [FusionModel] (Early Fusion -> 42-D Vector -> Random Forest Model)
       │
       ▼ (Engagement Score 1-5)
   [EngagementTracker] (Smooths values via EMA, saves record)
       │
       ▼ (Queue)
   [Dashboard App] (Drains queue & updates Streamlit UI)
```

---

## 7. Quality Audits & Architectural Smells

1. **Unused Imports and Constants**:
   - `src/core/constants.py` and `src/core/logger.py` are empty files containing no code. They are imported in several places or defined but never populated.
   - *Recommendation*: Clean up empty files or implement central logger handlers inside them.

2. **Diarization Granularity**:
   - `SpeakerDiarizer` runs online KMeans on a 3-second segment average. If both speakers talk during the same segment, the entire segment's transcript is incorrectly attributed to a single speaker.
   - *Recommendation*: Move to a frame-level diarization model or run VAD-based segmentation prior to feature extraction.

3. **Streamlit Polling Jitter**:
   - `app.py` sleeps for 1 second and executes `st.rerun()`. This causes high CPU usage and UI rendering lag on slower machines.
   - *Recommendation*: Refactor the production dashboard to a decoupled Next.js/React frontend communicating with a FastAPI backend over WebSockets.
