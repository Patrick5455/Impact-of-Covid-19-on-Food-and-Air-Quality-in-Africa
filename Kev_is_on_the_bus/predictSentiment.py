import pickle
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

KERAS_MODEL = "model.h5"
TOKENIZER_MODEL = "tokenizer.pkl"
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SEQUENCE_lENGTH = 300
SENTIMENT_THRESHOLDS = (0.4, 0.7)

def decodeSentiment(score, includeNeutral=True):
    if includeNeutral:
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predictText(text, model, tokeniza, includeNeutral=True, label=True):
    x_test = pad_sequences(tokeniza.texts_to_sequences([text]), maxlen=SEQUENCE_lENGTH)
    score = model.predict([x_test])[0]

    senti = decodeSentiment(score, includeNeutral)

    if label:
        return senti

    return score

if __name__ == "__main__":

    df = pd.read_csv('latestTotal.csv')
    df = df.loc[df.clean_text.apply(lambda x: not isinstance(x, (float, int)))]

    model = load_model(KERAS_MODEL)

    with open(TOKENIZER_MODEL, 'rb') as handle:
        tokeniza = pickle.load(handle)

    df['sentiment'] = df.clean_text.apply(lambda x: predictText(x, model, tokeniza))
    df['sentimentScore'] = df.clean_text.apply(lambda x: predictText(x, model, tokeniza, label=False))

    print(df.head(5))
    df.to_csv("SentimentLatestTotal.csv")
