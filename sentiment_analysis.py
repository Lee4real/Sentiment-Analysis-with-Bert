from transformers import BertForSequenceClassification, BertTokenizerFast, pipeline

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('finetuned_BERT_epoch_{epoch}.model')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Create a pipeline
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


def analyze_sentiment(text):
    result = nlp(text)[0]
    return result


if __name__ == "__main__":
    text = "Replace this with the text you want to analyze"
    sentiment = analyze_sentiment(text)
    print(sentiment)
