def load_classes():
    path = "C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\base_ontology\\classes.txt"
    with open(path, "r") as f:
        classes = [line.strip() for line in f if line.strip()]

        return classes