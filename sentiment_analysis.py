

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


##datasets
def create_dataset():
    """
    Create a labeled dataset of product reviews with sentiment labels.
    Returns a pandas DataFrame with 'review' and 'sentiment' columns.
    """
    reviews = [
        # Positive reviews in Complex language
        ("This product is amazing! I love it so much.", 1),
        ("Excellent quality and fast shipping.", 1),
        ("Best purchase I've made. Highly recommend!", 1),
        ("Great value for money. Very satisfied.", 1),
        ("Absolutely wonderful. Five stars!", 1),
        ("Perfect! Works as expected.", 1),
        ("Outstanding quality. Very impressed.", 1),
        ("Fantastic item! Exceeded all expectations.", 1),
        ("Superb product. Will buy again!", 1),
        ("Incredibly pleased with my order.", 1),
        ("This is exactly what I needed.", 1),
        ("Amazing value and excellent service.", 1),
        ("Top notch quality. Highly satisfied!", 1),
        ("Wonderful experience from start to finish.", 1),
        ("Best investment I made this year.", 1),
        ("Exceeded expectations in every way.", 1),
        
        # Positive reviews in Simple language
        ("Good product! I like it.", 1),
        ("Great! Very happy.", 1),
        ("Love it! So good.", 1),
        ("Good quality. Happy customer here.", 1),
        ("Love this! Great stuff.", 1),
        ("Good. Works great.", 1),
        ("I love it so much!", 1),
        ("Loving it! Really good.", 1),
        ("Great quality. Love it.", 1),
        ("Good product. Love the quality.", 1),
        ("Love it! Loving every moment.", 1),
        ("Great! Happy with it.", 1),
        ("Good! Very satisfied.", 1),
        ("Love the product. Good stuff.", 1),
        ("Great product. Loving it!", 1),
        ("Good and great quality!", 1),
        ("Love it. Good investment.", 1),
        ("Happy and loving this product.", 1),
        ("Good quality. Great service.", 1),
        ("Love it! Great find.", 1),
        
        # Negative reviews in Complex language
        ("This is terrible. Does not work at all.", 0),
        ("Very disappointed with this purchase.", 0),
        ("Poor quality. Waste of money.", 0),
        ("Worst product ever. Do not buy!", 0),
        ("Completely broken. Very unhappy.", 0),
        ("Awful experience. Not recommended.", 0),
        ("Horrible quality. Total waste.", 0),
        ("Does not match the description at all.", 0),
        ("Broke after just one week of use.", 0),
        ("Total disaster. Cannot return it.", 0),
        ("Worst purchase ever made online.", 0),
        ("Stay away from this product.", 0),
        ("Defective product. No refund offered.", 0),
        ("Regrets. Should never have bought this.", 0),
        ("Worst experience ever with this brand.", 0),
        
        # Negative reviews in Simple language
        ("Bad product. Not good.", 0),
        ("Bad quality. Hate it.", 0),
        ("Worse than expected. Bad.", 0),
        ("Bad product. Not worth it.", 0),
        ("Hate it. Bad quality.", 0),
        ("Very bad. Hate the product.", 0),
        ("Bad and broken. Hate it.", 0),
        ("Worse than anything. Bad choice.", 0),
        ("Bad quality. Hate buying this.", 0),
        ("Not good at all. Bad product.", 0),
    ]
    
    # DataFrame
    df = pd.DataFrame(reviews, columns=['review', 'sentiment'])
    print("Dataset created successfully!")
    print(f"Total samples: {len(df)}")
    print(f"Positive reviews: {(df['sentiment'] == 1).sum()}")
    print(f"Negative reviews: {(df['sentiment'] == 0).sum()}")
    print("\nFirst 3 reviews:")
    print(df.head(3))
    
    return df

##text preprossesing

def preprocess_text(text):
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def preprocess_dataset(df):
    df['review'] = df['review'].apply(preprocess_text)
    print("\n✓ Text preprocessing completed")
    return df


##feature extraction by tf-idf

def vectorize_text(X_train, X_test):
   
    # Creating vectorizer
    vectorizer = TfidfVectorizer(max_features=100, lowercase=True)
    
    # Fitting on training data and transforming both train and test
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"✓ TF-IDF vectorization completed")
    print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"  Training data shape: {X_train_vec.shape}")
    print(f"  Test data shape: {X_test_vec.shape}")
    
    return vectorizer, X_train_vec, X_test_vec


#splitting and training
def train_sentiment_model(X_train_vec, y_train):
    
    # Initialize and train the model
    model = LogisticRegression(max_iter=100, random_state=42)
    model.fit(X_train_vec, y_train)
    
    print("\n✓ Model training completed")
    print(f"  Algorithm now on running: Logistic Regression")
    
    return model

#model evaluation

def evaluate_model(model, X_test_vec, y_test):
   
    # Make predictions on test data
    y_pred = model.predict(X_test_vec)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Negative (0)', 'Positive (1)']))
    
    return accuracy, y_pred


# custom ny taking input from user

def predict_sentiment(text, model, vectorizer):
    
    # Preprocess the input text
    text_processed = preprocess_text(text)
    
    # Vectorize the text
    text_vec = vectorizer.transform([text_processed])
    
    # Make prediction
    sentiment = model.predict(text_vec)[0]
    
    # Get confidence score (probability)
    confidence = model.predict_proba(text_vec)[0]
    
    # Return sentiment and confidence
    sentiment_label = "POSITIVE ✓" if sentiment == 1 else "NEGATIVE ✗"
    
    return sentiment, sentiment_label, confidence

#main exec
def main():
    """Execute the complete sentiment analysis pipeline."""
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS PROJECT - BEGINNER LEVEL")
    print("="*60 + "\n")
    
    #Creating dataset
    print("STEP 1: Dataset Creation")
    print("-" * 60)
    df = create_dataset()
    
    #Preprocessing text
    print("\nSTEP 2: Text Preprocessing")
    print("-" * 60)
    df = preprocess_dataset(df)
    
    #Preparing features and labels
    X = df['review'].to_numpy()
    y = df['sentiment'].to_numpy()
    
    #Splitting dataset into training and test sets
    print("\nSTEP 3: Train-Test Split")
    print("-" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"✓ Data split completed")
    print(f"  Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")
    
    #vectorize text using TF-IDF
    print("\nSTEP 4: Feature Extraction (TF-IDF)")
    print("-" * 60)
    vectorizer, X_train_vec, X_test_vec = vectorize_text(X_train, X_test)
    
    #Training the model
    print("\nSTEP 5: Model Training")
    print("-" * 60)
    model = train_sentiment_model(X_train_vec, y_train)
    
    #evaluating the model
    print("\nSTEP 6: Model Evaluation")
    print("-" * 60)
    accuracy, predictions = evaluate_model(model, X_test_vec, y_test)
    
    #for Custom predictions
    print("\n" + "="*60)
    print("STEP 7: Custom Sentiment Prediction")
    print("="*60)
    
    #with some custom sentences
    test_sentences = [
        "I absolutely love this product!",
        "This is the worst thing I've ever bought.",
        "It's okay, nothing special.",
    ]
    
    print("\nTesting custom sentences:\n")
    for sentence in test_sentences:
        sentiment, label, confidence = predict_sentiment(sentence, model, vectorizer)
        print(f"Input: \"{sentence}\"")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence[sentiment]:.2%}\n")
    
    #interactive prediction
    print("="*60)
    print("Interactive Mode - Enter Your Own Review")
    print("="*60)
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("Enter a product review: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nThank you for using the Sentiment Analysis tool!")
            break
        
        if not user_input:
            print("Please enter a valid sentence.\n")
            continue
        
        sentiment, label, confidence = predict_sentiment(user_input, model, vectorizer)
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence[sentiment]:.2%}\n")


if __name__ == "__main__":
    main()
