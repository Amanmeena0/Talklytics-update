import numpy as np
from sklearn.cluster import MiniBatchKMeans

class SpeakerDiarizer:
    """
    Lightweight streaming speaker diarization using online KMeans clustering
    on acoustic feature vectors (MFCCs, Pitch, Energy).
    Assumes 2 speakers (e.g., Sales Rep and Prospect).
    """
    def __init__(self, n_speakers: int = 2):
        self.n_speakers = n_speakers
        # MiniBatchKMeans is perfect for streaming partial_fit
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_speakers, 
            random_state=42, 
            n_init=1,
            batch_size=10
        )
        self._is_initialized = False

    def identify_speaker(self, feature_vector: np.ndarray) -> str:
        """Identify speaker using online clustering of acoustic features."""
        X = feature_vector.reshape(1, -1)
        
        if not self._is_initialized:
            # Initialize with dummy distinct centroids to prevent ValueError
            # and set the feature dimensionality correctly.
            dummy_data = np.zeros((self.n_speakers, X.shape[1]))
            for i in range(self.n_speakers):
                dummy_data[i, :] = i * 2.0  # Separate the initial centroids
            self.kmeans.partial_fit(dummy_data)
            self._is_initialized = True

        # Predict the cluster for the current segment
        cluster_id = self.kmeans.predict(X)[0]
        
        # Update the centroids with the new data to adapt to the speakers
        self.kmeans.partial_fit(X)
        
        return f"Speaker {cluster_id + 1}"
