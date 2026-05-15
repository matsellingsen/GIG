## System Architecture
### Design Philosophy (Ground my choice/motivation in reason and references)
#### System level principles
- Sequential
- Deterministic
- Interpretable & transparent
- Traceable
- Core functionality; I.e. get the "axioms" (meaning atomic questions) stable. 

#### Agent level principles
- Decomposed
- single-task 

#### prompt level principles
- Domain agnostic
- Markdown headers
- Strict Json Schemas

#### Why Multi-Agent Instead of Monolithic
- Reduced prompt complexity
- Improved controllability and modularity
- Easier debugging and error isolation
- Aligns with cognitive decomposition principles

#### Reason-First, Answer-Second Strategy
- Agents must produce structured reasoning before generating an answer
- Prevents hallucination and enforces ontology-faithfulness


### System overview
#### High-Level pipeline overview
- Illustration of the full system pipeline
- Explanation of the two major subsystems:
  - Ontology Construction Module (offline)
  - Inference Module (online)

#### Data Flow and I/O Contracts
- Definition of strict input/output schemas between agents
- Explanation of deterministic behavior enforcement
- JSON validation rules and schema constraints

#### Decoupling of Construction and Inference
- Ontology is built once and reused
- Inference module never modifies the ontology
- Guarantees reproducibility and stability

### Ontology construction & cleanup module 
#### Overview
- Illustration of ontology-construction & cleanup pipeline

#### Pipeline go-through
- Walk through of what & how the components/agents process the data sequentially

#### Ontology Constraints and Guarantees
- No cycles
- Unique identifiers
- Consistent relation typing
- Deterministic ordering
- Closed-world vs open-world considerations


### Inference module 
#### Overview
- Illustration of inference-module component

#### Atomic Question Assumption
- Why atomicity is required
- How it simplifies reasoning
- How it enables deterministic mapping

#### Pipeline go-through
- Walk through of how the components/agents process the data sequentially

#### 

### Component details
#### Agents
- For each agent:
    - Purpose
    - Inputs
    - Outputs
    - JSON schema
    - System prompt
    - Deterministic constraints
    - Failure modes
    - Example input/output

#### Ontology Specification
- Description of Ontology parts; Why and how they are used
    - Base (class / axioms)
    - Relation-Types (Subclass / dataProperty / objectProperty)
    - Entity-types (Class / instance ) 

#### Why OWL-Like but Not Full OWL
- OWL expressivity vs system needs
- Trade-offs
- Rationale for a simplified ontology for deterministic LLM reasoning
    