# Prewords
Hello there! 
Welcome to the implementation of the GIG system, which was created by yours truly (Mats Ellingsen) as my Master Thesis. It is both a set of principles and also an implemented system. I know, confusing. Go read my thesis to wrap your mind around it! I don't know if its publically available anywhere though but maybe I'm smart at some point and figure that out. In the meantime, best of luck going through the project! I (copilot) have carefully created a helpful explanation at an attempt to untangle whats going on here. 


# Grounded Information Generation (GIG)

Grounded Information Generation (GIG) is the culmination of Mats Ellingsen's Master's thesis, introducing an ontology-grounded question-answering system designed to mitigate hallucinations in Large Language Models (LLMs) by enforcing strict factual reliability. By relying on a stable, pre-constructed Web Ontology Language (OWL) knowledge base and employing a decomposition-first multi-agent architecture, the system guarantees that all generated answers are explicitly grounded and formally traceable.

This repository contains the full implementation of the GIG system, serving as the technical proof of concept for the thesis. It includes the offline Knowledge-Base Construction subsystem and the online Inference subsystem, as well as the evaluation datasets and test scripts used to conduct the study's empirical evaluation.

## Project Structure

The repository is organized as follows:

```text
├── models/                     # Directory for storing the downloaded local models
├── src/system_v5/              # Main system source code
│   ├── agent_loop/             # Individual agent scripts, organized by functional domain
│   ├── backends/               # Model and hardware abstraction (OpenVINO NPU configuration)
│   ├── content/                # Source texts and chunked documents used for ontology construction
│   ├── KB/                     # Persistent Knowledge Base (finalized ontologies and version metadata)
│   ├── pipelines/              # High-level orchestration scripts for construction and inference
│   ├── prompts/                # Markdown-formatted system and user prompts for all agents
│   └── tools/                  # Utility functions and helper modules
├── src/system_v5/tests/        # Component and end-to-end evaluation scripts and datasets
├── .requirements.txt           # Python dependencies
└── todo.md                     # (very) personal and messy development notes 
```

## Hardware and Software Prerequisites

This system is designed for **local execution** to guarantee environmental control and data privacy. It leverages OpenVINO for hardware-accelerated inference.

### Hardware
*   **Processor:** Intel Core Ultra processor with an integrated NPU (e.g., Intel® Core™ Ultra 7 258V or similar).
*   **Memory:** At least 16 GB, preferably 32 GB RAM to comfortably handle the quantized models and data caching.

### Software Environment
*   **OS:** Windows 11 (or compatible Linux setups with properly configured Intel NPU drivers)
*   **Python:** Python 3.10+
*   **Dependencies:** Listed in `.requirements.txt`.

### Model Requirement
The system is built around strict deterministic generation using a quantized variant of the Microsoft Phi-4-mini model. You will need to download the Intel NPU-optimized OpenVINO variant (e.g., `AhtnaGlen/phi-4-mini-instruct-int4-sym-npu-ov`) and place it inside the `models/` directory for local execution.

---

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/matsellingsen/GIG.git
   cd GIG
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows Git Bash
   # or .venv\Scripts\activate.bat on Windows Cmd
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r .requirements.txt
   ```

4. **Prepare the models:**
   Ensure that the Intel-quantized OpenVINO model (`phi-4-mini-instruct`) is available in the `models/` directory (Available at AhtnaGlen/phi-4-mini-instruct-int4-sym-npu-ov on HuggingFace!).

---

## How to Run the System

The workflow is divided into two primary subsystems located in the `src/system_v5/pipelines/` directory: an offline phase that builds the ontology and an online phase that interprets queries against it.

### 1. Knowledge Base Construction (Offline Phase)

This phase transforms the raw source text chunks located in `content/` into a persistent, clean, and resolved OWL ontology (Turtle format).

*   **Step 0: Configure source text**
    ```bash
    If you want to test the system on any other source text, make sure to first clean and chunk it and store it under content/. You may resue the script in tools/preprocessing, or use a custom one.
    ```

*   **Step 1: Extract concepts and assert rules**
    ```bash
    python src/system_v5/pipelines/ontology_construction.py
    ```

*   **Step 2: Clean and resolve entities & classes (De-duplication)**
    ```bash
    python src/system_v5/pipelines/ontology_cleanup.py
    ```

*   **Step 3: Serialize to Turtle Format (.ttl)**
    ```bash
    python src/system_v5/pipelines/ontology_json_to_turtle.py
    ```
    *This generates the finalized Knowledge Base snapshot located in the `src/system_v5/KB/` directory.*

### 2. Grounded Inference (Online Phase)

Once the Knowledge Base is finalized, you can query the system. The inference pipeline resolves the question type, retrieves graph snippets computationally related to the entities posed, generates an answer, and finally performs context mapping to check its grounding. 

*   **Run the inference pipeline:**
    ```bash
    python src/system_v5/pipelines/inference_module.py
    ```
    *Note: During runtime, the system will apply deterministic decoding settings (`temperature=0.0`, fixed random seed, nucleus sampling disabled) and execute heavily via the integrated Intel NPU.*

---

## Evaluation and Tests

The `src/system_v5/tests/` directory contains all datasets and scripts required to reproduce the evaluation results documented in the thesis:

*   **Component tests:** Isolating the *Input Interpretation* and *Entity Resolution* modules against standardized adversarial data formulations.
*   **Ontology-Grounded End-to-End Evaluation:** Validating logic pipelines assuming perfect upstream domain formulation.
*   **Source-Grounded End-to-End Evaluation:** Comparing the entire GIG stack against baseline RAG structures, demonstrating empirical reduction of false positives at the cost of strict "abstain" thresholds.

You can inspect the evaluation JSON sets directly within the `dataset/` subfolder, and execute interpretation evaluation scripts in `scripts/interpret_results/`.

---

## Thesis Reference

For detailed reasoning on why Open World Assumption (OWA) was utilized, specifics on the constraints of Atomic Question formulation, or how the class-resolution agent pipeline was designed regarding descriptive logic (subsets of DOLCE+DnS Ultralite), please consult the accompanying Master's Thesis text.

## License

*(If applicable, indicate your open-source licensing here, e.g., MIT License)*
