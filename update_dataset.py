import json

data_updates = {
  "q01": {"expected_entity": "SENS motion®", "expected_object": None},
  "q02": {"expected_entity": "SENS motion® system", "expected_object": None},
  "q03": {"expected_entity": "SENS motion® activity sensor", "expected_object": "healthcare"},
  "q04": {"expected_entity": "SENS motion® – For Research platform", "expected_object": None},
  "q05": {"expected_entity": "SENS Innovation ApS", "expected_object": None},
  "q06": {"expected_entity": "SENS motion® Pilot Package", "expected_object": None},
  "q07": {"expected_entity": "Privacy Policy of SENS Innovation ApS", "expected_object": None},
  "q08": {"expected_entity": "SENS motion®", "expected_object": "certified medical device"},
  "q09": {"expected_entity": "SENS Innovation ApS", "expected_object": "developer of medical sensor products"},
  "q10": {"expected_entity": "SENS motion® activity sensor", "expected_object": "SENS motion® – For Healthcare solution"},
  "q11": {"expected_entity": "smartphone app", "expected_object": "SENS motion® – For Research package"},
  "q12": {"expected_entity": "SENS Innovation ApS", "expected_object": "data controller for all projects"},
  "q13": {"expected_entity": "Bispebjerg Hospital study", "expected_object": "SENS Innovation’s own clinical trials"},
  "q14": {"expected_entity": "motivational app", "expected_object": "SENS motion® – For Healthcare"},
  "q15": {"expected_entity": "SENS motion®", "expected_object": "physical activity"},
  "q16": {"expected_entity": "bedside tablet", "expected_object": "patients"},
  "q17": {"expected_entity": "healthcare professionals", "expected_object": "SENS motion® data"},
  "q18": {"expected_entity": "SENS motion® smartphone app", "expected_object": None},
  "q19": {"expected_entity": "SENS motion® web‑application", "expected_object": None},
  "q20": {"expected_entity": "SENS motion®", "expected_object": "heart rate"},
  "q21": {"expected_entity": "SENS motion® system", "expected_object": "patients"},
  "q22": {"expected_entity": "SENS motion® Pilot Package", "expected_object": "patients"},
  "q23": {"expected_entity": "Pilot Package", "expected_object": None},
  "q24": {"expected_entity": "SENS motion® Pilot Package", "expected_object": None},
  "q25": {"expected_entity": "SENS Innovation ApS", "expected_object": "CVR number"},
  "q26": {"expected_entity": "SENS Innovation ApS", "expected_object": "address"},
  "q27": {"expected_entity": "aggregated website statistics", "expected_object": "retention period"},
  "q28": {"expected_entity": "elderly patients", "expected_object": "clinical evidence section"},
  "q29": {"expected_entity": "SENS Innovation ApS", "expected_object": "team members"},
  "q30": {"expected_entity": "Kasper L. Lykkegaard", "expected_object": "SENS Innovation ApS"},
  "q31": {"expected_entity": "Morten Kjærgaard", "expected_object": "SENS Innovation team"},
  "q32": {"expected_entity": "SENS motion®", "expected_object": "Privacy Policy"},
  "q33": {"expected_entity": "DEMOS‑10", "expected_object": "SENS Innovation ApS"},
  "q34": {"expected_entity": "Lifetrack", "expected_object": "Privacy Policy"},
  "q35": {"expected_entity": "motivational app", "expected_object": "SENS motion® – For Healthcare package"},
  "q36": {"expected_entity": "SENS motion®", "expected_object": "other activity monitors"},
  "q37": {"expected_entity": "SENS motion®", "expected_object": "reference accelerometers"},
  "q38": {"expected_entity": "elderly patients", "expected_object": "younger patients"},
  "q39": {"expected_entity": "SENS motion® motivational app", "expected_object": "bedside tablet"},
  "q40": {"expected_entity": "SENS motion® – For Healthcare package", "expected_object": "SENS motion® – For Research package"},
  "q41": {"expected_entity": "SENS motion® system", "expected_object": "traditional patient monitoring methods"},
  "q42": {"expected_entity": "Pilot Package", "expected_object": "standard SENS motion® deployments"},
  "q43": {"expected_entity": "Bispebjerg Hospital", "expected_object": "out‑of‑bed time"},
  "q44": {"expected_entity": "elderly hospitalised patients", "expected_object": "inactive"},
  "q45": {"expected_entity": "steps", "expected_object": "readmission risk"},
  "q46": {"expected_entity": "elderly patients above age 65", "expected_object": "hospitalised"},
  "q47": {"expected_entity": "inactivity", "expected_object": "physical fitness"},
  "q48": {"expected_entity": "SENS motion® sensors", "expected_object": "web‑application"},
  "q49": {"expected_entity": "Pilot Package", "expected_object": "months"}
}

dataset_path = 'c:/Users/matse/gig/src/system_v5/tests/dataset/source_dataset.json'

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    q_id = item['id']
    if q_id in data_updates:
        item['expected_entity'] = data_updates[q_id]['expected_entity']
        item['expected_object'] = data_updates[q_id]['expected_object']

with open(dataset_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Updated dataset with expected_entity and expected_object labels.")
