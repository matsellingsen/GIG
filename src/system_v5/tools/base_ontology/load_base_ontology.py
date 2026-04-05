import json

def load_classes():
    path = "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\base_ontology\\DUL\\classes.txt"
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_classes = json.load(f)
            
        # Convert the JSON array into a {"ClassName": "Description"} Dictionary
        class_dict = {}
        for item in raw_classes:
            # Safely get the class name and description based on your JSON structure
            cls_name = item.get("class", "").strip()
            desc = item.get("description", "").strip()
            
            if cls_name:
                class_dict[cls_name] = desc
                
        return class_dict
        
    except Exception as e:
        print(f"Error loading base classes: {e}")
        return {}

def load_axioms():
    path = "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\base_ontology\\DUL\\axioms.txt"
    try:
        with open(path, "r", encoding="utf-8") as f:
            # Axioms can remain as a standard list of dictionaries
            axioms = json.load(f)
        return axioms
    except Exception as e:
        print(f"Error loading base axioms: {e}")
        return []