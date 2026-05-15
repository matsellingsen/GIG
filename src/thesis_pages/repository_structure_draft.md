#### Repository Structure

**High-Level Folder Hierarchy**
The core functionality of the system is contained within the most recent iteration folder, `src/system_v5/`. The highest level structure is intentionally segregated by functional responsibility—from environment setup down to the specific multi-agent logic.

A comprehensive view of all top-level directories within `src/system_v5/` includes:
```text
src/system_v5/
├── agent_loop/           # Individual single-task agents and their specific logic
├── backends/             # Code for interfacing with the LLM (e.g., OpenVINO, NPU)
├── content/              # Source texts, chunks, and documents for the system to process
├── KB/                   # Knowledge Base storage housing the built ontology 
├── pipelines/            # High-level orchestration scripts tying agents together
├── prompts/              # Domain-agnostic prompt templates
└── tools/                # Utility functions and external tools invoked by the system
```

**Purpose of Each Top-Level Directory**
* **`agent_loop/`**: Contains the decomposed agents. It acts as the operational brain of the system, taking structured data, reasoning over it, and passing the payload forward.
* **`backends/`**: Abstracts the model execution environment away from the reasoning loops. This ensures the inference hardware (e.g., NPU constraints or strict decoding settings) doesn’t bleed into agent logic.
* **`content/`**: Houses the raw and pre-processed source material, such as document chunks, that the system grounds its answers and ontology against.
* **`intermediate_results/`**: Serves as a working directory for saving temporary state and reasoning artifacts during multi-step pipeline executions for visibility and debugging.
* **`KB/`** (Knowledge Base): The persistent storage layer for the ontology state, segregating current, new, and archived ontology structures.
* **`pipelines/`**: Orchestrates the sequential execution of the system, handling macro-level connections between agents rather than the domain logic itself.
* **`prompts/`**: Stores domain-agnostic markdown prompt templates, separated from agent logic for easier configuration and tuning.
* **`tests/`**: Contains all artifacts required for validation, from unit tests evaluating individual chunks to manual annotation UIs (like Streamlit) for human-in-the-loop evaluation of systemic metrics.
* **`tools/`**: Holds functional capabilities or modular utilities that agents or pipelines can utilize during execution.

**Organization of Construction, Inference, and Backend Components**
The system rigidly separates the offline construction of the ontology from the online inference tasks referencing it.
* **Construction Components**: Found in `pipelines/ontology_construction.py` and `pipelines/ontology_cleanup.py`, which invoke specific construction agents inside `agent_loop/agents/ontology_construction/`.
* **Inference Components**: Housed under `pipelines/inference_module.py`. These consume the outputs of agents located in `agent_loop/agents/inference_module/`. The Inference Module strictly reads from the built ontology and is forbidden from making modifications to it.
* **Backend Components**: The runtime logic, including temperature configuration and token generation limits, is wrapped within `backends/`. 

**Storage of JSON Schemas, Prompts, and Ontology Files**
* **JSON Schemas**: Defining the strict I/O contracts, schemas are centralized in the `specifications/` directory. By storing them globally, both the offline construction modules and online inference modules are guaranteed to use synchronized structures.
* **Prompts**: Stored in the `prompts/` directory, these are kept domain-agnostic and leverage markdown headers to clearly signal task requirements and constraints to the LLM. 
* **Ontology Files**: Processed and finalized ontologies are handled in the `KB/` (Knowledge base) directory, reflecting the current state (`KB/current/`) of the world model limits.

**Test and Evaluation Scripts Structure**
The testing environment is isolated within the `tests/` directory to ensure reproducible and isolated testing routines:
* **`tests/scripts/`**: Houses utility scripts and evaluation dashboards (e.g., `test_source_dataset.py`, `manually_evaluate_source_results.py`). This includes Streamlit applications built for human-in-the-loop metric evaluation and caching strategies tying results directly to ground-truth text.
* **`tests/dataset/`**: The ground-truth material (`source_dataset.json`), representing curated inputs mapped closely to content chunks (`content/chunks.jsonl`).
* **`tests/reports/` and `tests/interpreted_results/`**: Execution artifacts, test metrics (like fuzz-matched text verification), and human-evaluated responses, strictly tracking the system’s performance.
