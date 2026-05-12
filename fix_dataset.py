import json
import re

with open('c:/Users/matse/gig/src/system_v5/tests/dataset/source_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

with open('c:/Users/matse/gig/src/system_v5/content/chunks.jsonl', 'r', encoding='utf-8') as f:
    chunks = {json.loads(line)['chunk_id']: json.loads(line)['chunk_text_clean'] for line in f}

corrections = {}

for item in dataset:
    q_id = item['id']
    if q_id == "q08":
        # Question: Is SENS motion a type of certified medical device?
        # Chunk text: Certified medical device (CE-marked)
        # Type: taxonomic (correct)
        pass
    if q_id == "q09":
        # Question: Is SENS Innovation ApS a medical device manufacturer?
        # Chunk text says "The company develops medical sensor products"
        # Let's change the question to strictly match the text
        corrections[q_id] = {
            "question": "Is SENS Innovation ApS a developer of medical sensor products?",
            "gold_answer": "yes, SENS Innovation ApS is a developer of medical sensor products.",
            "expected_object": "developer of medical sensor products",
            "question_type": "taxonomic"
        }
    if q_id == "q10":
        # Q: Is the SENS motion activity sensor a type of wearable medical sensor?
        # Text: Discrete Wearable Activity Sensor
        corrections[q_id] = {
            "question": "Is the SENS motion® activity sensor a discrete wearable activity sensor?",
            "gold_answer": "yes, the SENS motion® activity sensor is a discrete wearable activity sensor.",
            "expected_object": "discrete wearable activity sensor",
            "question_type": "taxonomic"
        }
    if q_id in ["q12", "q13"]:
        # Unanswerable taxonomic. No change needed since they are correctly unanswerable.
        pass

for item in dataset:
    q_id = item['id']
    if q_id in corrections:
        for k, v in corrections[q_id].items():
            item[k] = v
    if 'source' in item and isinstance(item['source'], str):
        item['source'] = item['source'].replace(')', '')

with open('c:/Users/matse/gig/src/system_v5/tests/dataset/source_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)
