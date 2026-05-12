import json
import sys

with open('c:/Users/matse/gig/src/system_v5/tests/dataset/source_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

chunks = {}
with open('c:/Users/matse/gig/src/system_v5/content/chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        c = json.loads(line)
        chunks[c['chunk_id']] = c.get('chunk_text_clean', '')

out = []
for item in dataset:
    if item['question_type'] == 'definition':
        continue
    source_id = item.get('source', '').replace(')', '') # some had stray parentheses
    res = f"ID: {item['id']}\nQ: {item['question']}\nType: {item['question_type']}\nAns: {item['gold_answer']}\nSource: {source_id}\nExists: {source_id in chunks}\n"
    if source_id in chunks:
        res += f"Chunk text: {chunks[source_id][:200]}...\n"
    res += "-" * 40
    out.append(res)

with open('check_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
