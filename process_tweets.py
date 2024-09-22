# process_tweets.py

import json
import bz2
import os
import pickle as pkl
from utils.data_processing import lexicon_dict, allcountrylist, famouscountry
from utils.tweet_filters import locationfilter, categoryfilter, languagefilter

def process_tweets(tweet_files, csvpath, output_path):
    lexdict = lexicon_dict(csvpath)
    country = allcountrylist()
    allnameslist = famouscountry(csvpath, country)
    cat_dict = {'Depression': [], 'Addiction': [], 'Anxiety': []}

    for file_name in tweet_files:
        with bz2.open(file_name, "rt") as bzinput:
            for line in bzinput:
                try:
                    tweet = json.loads(line)
                    if languagefilter(tweet):
                        text = tweet.get('retweetedStatus', {}).get('text', tweet.get('text', ''))
                        if locationfilter(allnameslist, text):
                            for cat in cat_dict.keys():
                                if categoryfilter(lexdict[cat], text) and len(cat_dict[cat]) < 400000:
                                    cat_dict[cat].append(text)
                except json.JSONDecodeError:
                    continue

    pkl.dump(cat_dict, open(os.path.join(output_path, "filtered_tweets.pkl"), "wb"))

    for cat in cat_dict:
        print(f"{cat}: {len(cat_dict[cat])}")
