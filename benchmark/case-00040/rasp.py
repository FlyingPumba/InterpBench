from tracr.rasp import rasp


def get_program() -> rasp.SOp:
  return make_emoji_sentiment_classifier(rasp.tokens)

def make_emoji_sentiment_classifier(sop: rasp.SOp) -> rasp.SOp:
    """
    Classifies each token as 'positive', 'negative', or 'neutral' based on emojis.

    Example usage:
      emoji_sentiment = make_emoji_sentiment_classifier(rasp.tokens)
      emoji_sentiment(["😊", "😢", "📘"])
      >> ["positive", "negative", "neutral"]
    """
    # Define mapping for emoji sentiment classification
    emoji_sentiments = {"😊": "positive", "😢": "negative", "📘": "neutral"}
    classify_sentiment = rasp.Map(lambda x: emoji_sentiments.get(x, "neutral"), sop)
    return classify_sentiment