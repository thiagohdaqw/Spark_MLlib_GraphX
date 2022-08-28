import sys
import re

DATASET = sys.argv[1]
CLEAN_REGEX = r"(</?\s*\w*\s*/?>)|[~!@#$?:>;,\.\-\\\"]+"

sentiments = {
    "negative": 0.0,
    "positive": 1.0
}

with open(DATASET, "r") as d, open(f"clean_{DATASET}", "w") as f:
    for lines in d:
        sentence, sentiment = lines.rsplit(",", 1)
        sentence_cleaned = re.sub(CLEAN_REGEX, '', sentence).lower()

        sentiment_cleaned = sentiment.strip()
        sentiment_number = sentiments.get(sentiment_cleaned, sentiment_cleaned)

        f.write(f"{sentence_cleaned};{sentiment_number}\n")

