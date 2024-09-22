# run_processing.py

from process_tweets import process_tweets

def main():
    csvpath = "/path/to/CSVFiles/"
    output_path = "/path/to/Output/"
    tweet_files = ["/path/to/data/" + file for file in os.listdir("/path/to/data/") if file.endswith(".json.bz2")]
    process_tweets(tweet_files, csvpath, output_path)

if __name__ == "__main__":
    main()
