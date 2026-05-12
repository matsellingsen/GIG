import json

with open('c:/Users/matse/gig/src/system_v5/tests/dataset/source_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Q11: taxonomic -> membership
# Q13: taxonomic -> membership
# Q14: taxonomic -> membership
# Q22: property -> quantification
# Q23: property -> quantification
# Q24: property -> quantification
# Q28: property -> quantification

updates = {
    "q11": {"question_type": "membership"},
    "q13": {"question_type": "membership"},
    "q14": {"question_type": "membership"},
    "q22": {"question_type": "quantification"},
    "q23": {"question_type": "quantification"},
    "q24": {"question_type": "quantification"},
    "q28": {"question_type": "quantification"}
}

for item in dataset:
    q_id = item['id']
    if q_id in updates:
        for k, v in updates[q_id].items():
            item[k] = v

with open('c:/Users/matse/gig/src/system_v5/tests/dataset/source_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)
