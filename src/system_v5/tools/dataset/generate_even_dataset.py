import json
from pathlib import Path

# Rules:
# definition -> [value]
# taxonomic -> [list, assertion]
# property -> [value, list, assertion]
# membership -> [list]
# capability -> [assertion, value]
# comparative -> [assertion, value]
# quantification -> [value]
# unknown -> [assertion, value, list]

# We will generate one canonical, paraphrased, and adversarial example for each domain, for each answer form of each question type.

domains = {
    "biology": {
        "definition": {
            "value": [
                ("What is a Cell?", "Describe what a Cell is.", "So what exactly is a Cell supposed to be?"),
                "be", {"type": "Class", "value": "Cell"}, {"type": null, "value": None}
            ]
        },
        "taxonomic": {
            "assertion": [
                ("Is a Mammal a type of Animal?", "Would you classify a Mammal as an Animal?", "Isn't it true that a Mammal isn't really an Animal?"),
                "be subtype of", {"type": "Class", "value": "Mammal"}, {"type": "Class", "value": "Animal"}
            ],
            "list": [
                ("What are the subtypes of Animal?", "List the types that fall under Animal.", "Name every single subtype of Animal supposedly existing."),
                "be subtype of", {"type": "Class", "value": "Animal"}, {"type": null, "value": None}
            ]
        },
        "property": {
            "value": [
                ("What is the color of LeafA?", "What property does LeafA have regarding its color?", "What color is LeafA actually supposed to have?"),
                "have property", {"type": "Individual", "value": "LeafA"}, {"type": "Literal", "value": "color"}
            ],
            "list": [
                ("What properties does LeafA have?", "List the attributes describing LeafA.", "Tell me every property LeafA supposedly possesses."),
                "have property", {"type": "Individual", "value": "LeafA"}, {"type": null, "value": None}
            ],
            "assertion": [
                ("Does LeafA have the property of being green?", "Is green a property of LeafA?", "Does LeafA actually even have the property green?"),
                "have property", {"type": "Individual", "value": "LeafA"}, {"type": "Literal", "value": "green"}
            ]
        },
        "membership": {
            "list": [
                ("What components are found in a PlantCell?", "Which structures are included in a PlantCell?", "Name all the things a PlantCell supposedly contains."),
                "have member", {"type": "Class", "value": "PlantCell"}, {"type": null, "value": None}
            ]
        },
        "capability": {
            "assertion": [
                ("Can a Neuron detect signals?", "Does a Neuron use signals?", "Can a Neuron supposedly process anything meaningful like signals?"),
                "have capability", {"type": "Class", "value": "Neuron"}, {"type": "Class", "value": "signals"}
            ],
            "value": [
                ("What does a Ribosome produce?", "What is produced by a Ribosome?", "What exactly does a Ribosome even produce then?"),
                "have capability", {"type": "Class", "value": "Ribosome"}, {"type": null, "value": None}
            ]
        },
        "comparative": {
            "assertion": [
                ("Is CellA greater than CellB in size?", "Would you say CellA is larger than CellB?", "Is CellA really even bigger than CellB?"),
                "compare", {"type": "Individual", "value": "CellA"}, {"type": "Individual", "value": "CellB"}
            ],
            "value": [
                ("Which is larger, CellA or CellB?", "Between CellA and CellB, which has a greater size?", "Who is really larger, CellA or CellB?"),
                "compare", {"type": "Individual", "value": "CellA"}, {"type": "Individual", "value": "CellB"}
            ]
        },
        "quantification": {
            "value": [
                ("How many chromosomes does a Human have?", "What is the count of chromosomes in a Human?", "Exactly how many chromosomes is a Human supposed to have?"),
                "count", {"type": "Class", "value": "Human"}, {"type": "Class", "value": "chromosomes"}
            ]
        },
        "unknown": {
            "assertion": [
                ("Is the answer 42?", "Would you say it is 42?", "Is it even remotely close to 42?"),
                "unknown", {"type": "Unknown", "value": "answer"}, {"type": "Unknown", "value": "42"}
            ],
            "value": [
                ("What happens?", "Describe what occurs.", "What exactly is supposed to happen?"),
                "unknown", {"type": "Unknown", "value": "event"}, {"type": null, "value": None}
            ],
            "list": [
                ("What are the things?", "List the things.", "Name all the random things."),
                "unknown", {"type": "Unknown", "value": "things"}, {"type": null, "value": None}
            ]
        }
    }
}

# Generalize the biology template via a simple dictionary rewrite for other domains
domain_subjects = {
    "biology": {"Cell": "Cell", "Mammal": "Mammal", "Animal": "Animal", "LeafA": "LeafA", "PlantCell": "PlantCell", "Neuron": "Neuron", "signals": "signals", "Ribosome": "Ribosome", "CellA": "CellA", "CellB": "CellB", "Human": "Human", "chromosomes": "chromosomes"},
    "geography": {"Cell": "Mountain", "Mammal": "River", "Animal": "Waterbody", "LeafA": "PeakA", "PlantCell": "Continent", "Neuron": "Satellite", "signals": "storms", "Ribosome": "Volcano", "CellA": "LakeA", "CellB": "LakeB", "Human": "Planet", "chromosomes": "oceans"},
    "technology": {"Cell": "Processor", "Mammal": "Laptop", "Animal": "Computer", "LeafA": "DeviceA", "PlantCell": "Server", "Neuron": "Sensor", "signals": "temperature", "Ribosome": "Compiler", "CellA": "NodeA", "CellB": "NodeB", "Human": "Network", "chromosomes": "nodes"},
    "people": {"Cell": "Person", "Mammal": "Teacher", "Animal": "Profession", "LeafA": "Alice", "PlantCell": "Family", "Neuron": "Doctor", "signals": "symptoms", "Ribosome": "Chef", "CellA": "Bob", "CellB": "Charlie", "Human": "Team", "chromosomes": "members"},
    "abstract": {"Cell": "Concept", "Mammal": "Honesty", "Animal": "Virtue", "LeafA": "IdeaA", "PlantCell": "Framework", "Neuron": "Algorithm", "signals": "patterns", "Ribosome": "Model", "CellA": "TheoryA", "CellB": "TheoryB", "Human": "Paradigm", "chromosomes": "principles"},
    "everyday": {"Cell": "Chair", "Mammal": "Mug", "Animal": "Container", "LeafA": "ObjectA", "PlantCell": "Backpack", "Neuron": "Thermometer", "signals": "heat", "Ribosome": "CoffeeMachine", "CellA": "BoxA", "CellB": "BoxB", "Human": "Drawer", "chromosomes": "items"}
}

out_data = {}
for dom, mapping in domain_subjects.items():
    out_data[dom] = {"canonical": [], "paraphrased": [], "adversarial": []}
    for qtype, forms in domains["biology"].items():
        for aform, payload in forms.items():
            inputs, rel, ent, obj = payload
            # Replace biology nouns with domain nouns
            def sub(text):
                if text is None: return text
                res = text
                for bio, dom_noun in mapping.items():
                    res = res.replace(bio, dom_noun)
                return res
                
            for idx, split in enumerate(["canonical", "paraphrased", "adversarial"]):
                new_ent = {"type": ent["type"], "value": sub(ent["value"])}
                new_obj = {"type": obj["type"], "value": sub(obj["value"])}
                
                # Rule enforces: definition, unknown, quantification -> null object
                # list answer forms -> null object
                # capability + value -> null object
                if (qtype in ["definition", "unknown", "quantification"]) or (aform == "list") or (qtype == "capability" and aform == "value"):
                     new_obj = {"value": None, "type": None}
                
                out_data[dom][split].append({
                    "atomic_input": sub(inputs[idx]).replace("?", ""),
                    "question_type": qtype,
                    "answer_form": aform,
                    "entity": new_ent,
                    "relation": rel,
                    "object": new_obj
                })

out_path = Path("c:/Users/matse/gig/src/system_v5/tests/dataset/synthetic_labelled_input.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)

print("Regenerated evenly distributed database successfully!")
