## Methodology
- When manually evaluating, two complementary principles were followed:
    - (1) Ontology‑grounded reasoning under the Open World Assumption (OWA)
    - (2) Context‑faithful reasoning for RAG‑based answers.
       - I.e. answer assertions that are not explicitly supported by the ground truth is deemed wrong.


## Terminology
### Precise 
- The answer expresses the correct meaning using accurate, specific, and semantically appropriate wording, without introducing semantic drift.

### Complete
- Answer contains all essential information to answer the question.

### Ground truth
- the source


## metrics
### Correct by ground truth 
- consistent with the ground truth
- consistent with the retrieved context
- Answers the question precisely 
- Answers the question completely

### Incomplete 
- consistent with the ground truth
- Answers the question precisely 
- Answer the question incompletely

### Unprecise 
- consistent with the ground truth
- Answers the question unprecisely  

### Incorrect (false information)
- States false information that is not consistent with the ground truth

### Incorrect (vague information)
- Not strictly incorrect, but extremely vague and not informational

### Incorrect Abstain 
- Abstains when answer is answerable based on ground truth

### Correct Abstain 
- Abstains when answer is unanswerable based on ground truth