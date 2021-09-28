import json

with open("query_result.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['title'])
    labels.append(item['label'])