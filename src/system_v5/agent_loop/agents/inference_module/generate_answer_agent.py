from ..base_ontology_agent import BaseOntologyAgent
from tools.load_prompt import load_prompt
import json
import os

class GenerateAnswerAgent(BaseOntologyAgent):
    def __init__(self, backend):
        super().__init__(backend=backend, system_prompt=None)

    def run(self, question_info, relevant_info):

        # 1. Fetch correct system prompt depending on question type
        question_type = question_info.get("question_type", "unknown")
        prompt_path = (
            f"C:\\Users\\matse\\gig\\src\\system_v5\\prompts\\system\\agents\\"
            f"inference_module\\generate_answer\\{question_type}.txt"
        )
        self.system_prompt = load_prompt(prompt_path)

        # 2. Build user message dynamically based on question type
        user_msg = self._build_user_message(question_type, question_info, relevant_info)

        # 3. Define answer schema
        answer_schema = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "answer": {"type": "string"}
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False
        }

        return self.generate_with_schema(user_msg, answer_schema)

    # ---------------------------------------------------------
    # Dynamic user message builder
    # ---------------------------------------------------------
    def _build_user_message(self, question_type, question_info, relevant_info):

        atomic_question = question_info.get("atomic_question", "unknown")
        question_info = {k: v for k, v in question_info.items() if k != "atomic_question"}  # drop atomic question from question info since it's already included separately in the prompt
        # Always include question info
        qinfo_str = json.dumps(question_info, indent=2)
        context_str = json.dumps(relevant_info, indent=2) if relevant_info else "None"

        # --- Definition questions ---
        if question_type == "definition":
            return f"""
            ### Task
            Answer a definition-style question. Describe what the entity *is* using only the provided context.

            ### Atomic Question
            {atomic_question}

            ### Question Information
            {qinfo_str}

            ### Relevant Ontology Context
            {context_str}
            """

        # --- Taxonomic questions ---
        if question_type == "taxonomic":
            return f"""
            ### Task
            Answer a taxonomic question about class membership or subclass relations.

            ### Atomic Question
            {atomic_question}

            ### Question Information
            {qinfo_str}

            ### Relevant Ontology Context
            {context_str}
            """
        
        # --- Capability questions ---
        if question_type == "capability":
            return f"""
            ### Task
            Answer a capability question about what the entity does or can do.
            
            ### Atomic Question
            {atomic_question}

            ### Question Information
            {qinfo_str}

            ### Relevant Ontology Context
            {context_str}
            """

        # --- Property questions ---
        if question_type == "property":
            return f"""
            ### Task
            Answer a property question about qualities, attributes, or literal values.
            
            ### Atomic Question
            {atomic_question}

            ### Question Information
            {qinfo_str}

            ### Relevant Ontology Context
            {context_str}
            """

        # --- Membership questions ---
        if question_type == "membership":
            return f"""
            ### Task
            Answer a membership question about parts, members, or included items.
            ### Atomic Question
            {atomic_question}

            ### Question Information
            {qinfo_str}

            ### Relevant Ontology Context
            {context_str}
            """

        # --- Comparative questions ---
        if question_type == "comparative":
            return f"""
            ### Task
            Answer a comparative question between two entities.
            ### Atomic Question
            {atomic_question}

            ### Question Information
            {qinfo_str}

            ### Relevant Ontology Context
            {context_str}
            """

        # --- Quantification questions ---
        if question_type == "quantification":
            return f"""
            ### Task
            Answer a quantification question about counts or quantities.

            ### Atomic Question
            {atomic_question}

            ### Question Information
            {qinfo_str}

            ### Relevant Ontology Context
            {context_str}
            """

        # --- Existential questions ---
        if question_type == "existential":
            return f"""
            ### Task
            Answer an existential question (yes/no) about whether a property or relation exists.

            ### Atomic Question
            {atomic_question}

            ### Question Information
            {qinfo_str}

            ### Relevant Ontology Context
            {context_str}
            """

        # --- Unknown / fallback ---
        return f"""
        ### Task
        The question type is unknown. Provide the best possible answer using only the provided context.

        ### Question Information
        {qinfo_str}

        ### Relevant Ontology Context
        {context_str}
        """

