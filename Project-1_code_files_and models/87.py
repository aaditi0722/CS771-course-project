import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.decomposition import PCA
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



# these are dummy models
class MLModel():
    def __init__(self) -> None:
        pass
    
    def train(self, X, y):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def predict(self, X):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
class TextSeqModel(MLModel):
    def __init__(self, model_path, max_length=50):
        super().__init__()
        self.model = self.load_saved_model(model_path)
        self.max_length = max_length

    def load_saved_model(self, model_path):
        # Load the trained model from the specified path
        return load_model("models/TextSeqModel.h5")

    def preprocess_input(self, X):
        # Tokenize and pad sequences
        tokenizer = Tokenizer(char_level=True)  # Character-level tokenizer
        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        return padded_sequences

    def predict(self, X):
        # Preprocess input data
        X_preprocessed = self.preprocess_input(X)
        
        # Add an additional dimension to match the model's expected input shape
        X_preprocessed = X_preprocessed[..., np.newaxis]  # Reshape to (batch_size, max_length, 1)
        
        # Predict using the loaded model
        y_pred = (self.model.predict(X_preprocessed) > 0.5).astype("int32")
        return y_pred
    
    
class EmoticonModel(MLModel):
    def __init__(self, model_path, max_length=13):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = Tokenizer(char_level=True)  # Character-level tokenizer
        self.cnn_model = self.load_cnn_model(model_path)  # Load the CNN model

    def load_cnn_model(self, model_path):
        """
        Load the saved CNN model from the specified path.
        
        Args:
        model_path (str): Path to the saved CNN model file.
        
        Returns:
        Model: Loaded CNN model.
        """
        return load_model(model_path)

    def preprocess_input(self, X):
        """
        Preprocess input data by tokenizing and padding the sequences.
        
        Args:
        X (list): List of emoticon strings.
        
        Returns:
        np.ndarray: Padded sequences.
        """
        self.tokenizer.fit_on_texts(X)  # Fit the tokenizer on the input data
        sequences = self.tokenizer.texts_to_sequences(X)  # Convert to sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        return padded_sequences

    def predict(self, X):
        """
        Predict labels using the loaded CNN model.
        
        Args:
        X (list): List of emoticon strings.
        
        Returns:
        np.ndarray: Predicted labels.
        """
        X_preprocessed = self.preprocess_input(X)  # Preprocess input data
        X_preprocessed = X_preprocessed[..., np.newaxis]  # Reshape to (batch_size, max_length, 1)
        
        # Predict using the loaded CNN model
        y_pred = (self.cnn_model.predict(X_preprocessed) > 0.5).astype("int32")
        return y_pred
    
class FeatureModel(MLModel):
    def __init__(self, model_path, n_components=50):
        super().__init__()
        self.model = load_model("models/FeatureModel.h5")
        self.pca = PCA(n_components=n_components)

    def predict(self, X):
        # Apply PCA transformation
        n_samples, n_time_steps, n_features = X.shape
        X_flattened = X.reshape(n_samples * n_time_steps, n_features)
        X_reduced_flat = self.pca.fit_transform(X_flattened)
        X_reduced = X_reduced_flat.reshape(n_samples, n_time_steps, self.pca.n_components)
        
        # Predict using the loaded model
        y_pred = (self.model.predict(X_reduced) > 0.5).astype("int32")
        return y_pred
    
class CombinedModel(MLModel):
    def __init__(self) -> None:
        train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
        train_feat_X = train_feat['features']
        train_feat_Y = train_feat['label']

        n = train_feat_X.shape[0]

        train_feat_X = train_feat_X[0:n]
        train_feat_Y = train_feat_Y[0:n]

        train_seq_df = pd.read_csv("datasets/train/train_emoticon.csv")

        train_e_X = train_seq_df['input_emoticon'].tolist()
        train_e_X = [[ord(x)  for x in e] for e in train_e_X]
        train_e_Y = train_seq_df['label'].tolist()

        X_train = np.array(train_e_X, dtype='float64')
        y_train =np.array(train_e_Y, dtype='float64')

        m = X_train.shape[0]
        m = int(m)

        X_train = X_train[0:m]
        y_train = y_train[0:m]

        m = train_feat_X.shape[0]
        m = int(m)
        train_feat_x = train_feat_X[0:m]

        X_train = X_train.reshape(X_train.shape[0], -1)

        X_train2 = train_feat_X.reshape(train_feat_X.shape[0], -1)

        X_train = np.concatenate((X_train, X_train2), axis=1)

        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))

        self.model.fit(X_train, y_train)

    def predict(self, X1, X2, X3): 
        X2 = [[ord(x) for x in e] for e in X2]
        X2 = np.array(X2, dtype='float64')

        X1 = X1.reshape(X1.shape[0], -1)
        X2 = X2.reshape(X2.shape[0], -1)
        
        X = np.concatenate((X2, X1), axis=1)
        return self.model.predict(X)
    
    
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{(pred)}\n")


if __name__ == '__main__':
    # read datasets
    test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']
    test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
    test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()
    
    # your trained models 
    feature_model = FeatureModel(model_path='models/FeatureModel.h5', n_components=50)
    text_model = TextSeqModel(model_path='models/TextSeqModel.h5')
    emoticon_model = EmoticonModel(model_path='models/EmoticonModel.h5', max_length=13)
    best_model = CombinedModel()
    
    # predictions from your trained models
    pred_feat = feature_model.predict(test_feat_X)
    pred_emoticons = emoticon_model.predict(test_emoticon_X)
    pred_text = text_model.predict(test_seq_X)
    pred_combined = best_model.predict(test_feat_X, test_emoticon_X, test_seq_X)
    
    # saving prediction to text files
    save_predictions_to_file(pred_feat, "pred_feat.txt")
    save_predictions_to_file(pred_emoticons, "pred_emoticon.txt")
    save_predictions_to_file(pred_text, "pred_text.txt")
    save_predictions_to_file(pred_combined, "pred_combined.txt")
    
    
