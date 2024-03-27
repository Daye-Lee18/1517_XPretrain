import torch

from nrclex import NRCLex


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

def encode_query(query, device):
    """ Create emotion columns using NRC lexicon
        : anger, anticipation, disgust, fear, joy, sadness, surprise, trust
    """
    emotion_count = NRCLex(query).raw_emotion_scores
    # shape: 8 (emotions)
    # encoded_query = torch.zeros(len(emostion_list), dtype=torch.long, device=device)
    encoded_query = torch.zeros(len(emostion_list), dtype=torch.long)
    for i, emo in enumerate(emostion_list):
        if emo in emotion_count.keys():
            encoded_query[i] = emotion_count[emo]
    return encoded_query.tolist()