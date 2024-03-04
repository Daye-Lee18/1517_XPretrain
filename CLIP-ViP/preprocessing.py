import argparse
import json
import nltk
import pandas as pd
import numpy as np

from tqdm import tqdm
from nrclex import NRCLex
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


emostion_list = [
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation"
]


def extract_emotion_manually(df, text=None):
    """ Extract the emotional data manually
    """
    # assign the default text
    if text is None:
        text = "happy|sad|afraid|fear|surprise|joy|disgust|annoy| anger|angry|" \
                "excite|excited|exciting|scare|scared|scary|fright|frighten|frightened|frightening" \
                "|fearful|fearless|fearfully"
    # extract the data that contains the text
    manual_df = df[df['caption'].str.contains(text)]
    print("Number of manually selecting data:", len(manual_df))
    return manual_df


def extract_emotional_data(df, sent_bound=0.6, manual=False):
    """ Extract the emotional data using the sentiment analysis
    """
    # Initialize SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    emotion_df = pd.DataFrame()

    # extract the emotional data using the sentiment analysis
    for idx in tqdm(range(len(df))):
        sentence = df.caption[idx]
        # calculate the sentiment score
        sentiment_score = sia.polarity_scores(sentence)
        # determine the emotional data: compound score > 0.6 (positive) or < -0.6 (negative)
        if sentiment_score['compound'] > sent_bound or sentiment_score['compound'] < -sent_bound:
            emotion_df = pd.concat([emotion_df, df.iloc[idx:idx+1, :]])

    print("Number of emotional data:", len(emotion_df))

    # extract the data that contains predefined emotion words
    if manual:
        manual_df  = extract_emotion_manually(df)
        # extract the data that is not in the emotion_df
        if "sen_id" in manual_df.columns:
            only_manual_df = pd.merge(manual_df, emotion_df, on='sen_id', how='outer', indicator=True).query('_merge=="left_only"')
            only_manual_df = only_manual_df.drop(columns=["caption_y", "video_id_y", "_merge"]).rename(columns={"caption_x": "caption", "video_id_x": "video_id"})
        else:
            only_manual_df = pd.merge(manual_df, emotion_df, on='video_id', how='outer', indicator=True).query('_merge=="left_only"')
            only_manual_df = only_manual_df.drop(columns=["caption_y", "_merge"]).rename(columns={"caption_x": "caption"})
        print("Number of data only in manual data:", len(only_manual_df))
        emotion_df = pd.concat([emotion_df, only_manual_df])
        print("Toal number of emotional data:", len(emotion_df))
        return emotion_df

    return emotion_df


def create_sentiment_columns(df):
    """ Create sentiment columns using NRC lexicon
        : positive, negative, neutral
    """
    df = df.reset_index(drop=True)
    for idx in tqdm(range(df.shape[0])):
        emotion_counts = NRCLex(df.caption.iloc[idx]).raw_emotion_scores
        # positive case
        if 'positive' in emotion_counts.keys():
            df.loc[idx, 'positive'] = emotion_counts['positive']
        # negative case
        if 'negative' in emotion_counts.keys():
            df.loc[idx, 'negative'] = emotion_counts['negative']
        # neutral case
        if 'positive' not in emotion_counts.keys() and 'negative' not in emotion_counts.keys():
            df.loc[idx, 'neutral'] = 1

    df.fillna({'positive': 0, 'negative': 0, 'neutral': 0}, inplace=True)
    return df


def create_emotion_columns(df):
    """ Create emotion columns using NRC lexicon
        : anger, anticipation, disgust, fear, joy, sadness, surprise, trust
    """
    df = df.reset_index(drop=True)
    for idx in tqdm(range(df.shape[0])):
        emotion_counts = NRCLex(df.caption.iloc[idx]).raw_emotion_scores
        # create the emotion columns
        for emo in emostion_list:
            if emo in emotion_counts.keys():
                df.loc[idx, emo] = emotion_counts[emo]

    df.fillna({emo: 0 for emo in emostion_list}, inplace=True)
    return df


def run_preprocessing():
    parser = argparse.ArgumentParser(description='Preprocess the data')
    parser.add_argument('--data_dir', type=str, help='data directory',
                        default='./data/msrvtt/annotations/msrvtt_test1k.json')
    parser.add_argument('--sent_bound', type=float, help='sentiment bound', default=0.6)    
    parser.add_argument('--output_dir', type=str, help='output directory', default='./data/msrvtt/annotations/')
    args = parser.parse_args()
    
    # Load the data
    with open(args.data_dir, 'r') as f:
        data_json = json.load(f)
    df = pd.DataFrame(data_json)
    df['video_id'] = df['video'].apply(lambda x: x.split('.')[0])
    print("Number of data:", len(df))

    # Group the data by video_id
    merge_caption = lambda x: '. '.join(x)
    video_df = df.groupby('video_id').aggregate('caption').apply(merge_caption)
    video_df = pd.DataFrame(video_df).reset_index()

    # Extract the emotional data
    emotion_df = extract_emotional_data(video_df, sent_bound=args.sent_bound, manual=True)
    
    # Extract the data without emotion
    no_emotion_df = video_df[~video_df['video_id'].isin(emotion_df['video_id'])]
    num_emotion, num_no_emotion = len(emotion_df), len(no_emotion_df)
    print("Number of data without emotional data:", len(no_emotion_df))

    # Balance the data
    num_min = min(num_emotion, num_no_emotion)
    emotion_df = emotion_df.sample(n=num_min, random_state=42)
    no_emotion_df = no_emotion_df.sample(n=num_min, random_state=42)
    total_df = pd.concat([emotion_df, no_emotion_df])
    df = df[df['video_id'].isin(total_df['video_id'])]

    # create the sentiment and emotion columns
    df = create_sentiment_columns(df)
    df = create_emotion_columns(df)

    # save the data
    output_file = args.output_dir + 'emotion_test_df'
    df.to_csv(output_file + '.csv', index=False)
    df.to_json(output_file + '.json', orient='records', lines=False)
    print(f"Data is saved in {output_file}.csv and {output_file}.json")

if __name__ == "__main__":
    run_preprocessing()