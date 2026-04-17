from ..ontology_construction.base_construction_agent import BaseConstructionAgent
from tools.load_prompt import load_prompt

class SemanticClusterAgent(BaseConstructionAgent):
    def __init__(self, backend):
        # Load the corresponding system prompt
        system_prompt = load_prompt(r"c:\Users\matse\gig\src\system_v5\prompts\system\agents\ontology_cleanup\semantic-cluster.txt")
        super().__init__(backend=backend, system_prompt=system_prompt)

    def run(self, cluster: list, cluster_context: dict = None) -> tuple:
        """
        Runs the LLM on a Fuzzy Semantic Cluster (instances with high string similarity).
        Args:
            cluster (list): A list of instance name strings that are highly similar.
            cluster_context (dict, optional): A dictionary keyed by instance name with its 'class',
                                              'class_description' and 'properties' lists.
        Returns:
            dict: Mapping of original string to the canonical string it should be merged into.
            str: The prompt used.
        """
        if not cluster or len(cluster) < 2:
            return {}, "Skipped: Cluster too small."

        # Format input for the model
        cluster_block = ""
        for name in cluster:
            cluster_block += f"- Name: {name}\n"
            if cluster_context and name in cluster_context:
                ctx = cluster_context[name]
                cls_name = ctx.get("class", "Unknown")
                cls_desc = ctx.get("class_description", "Unknown")
                props = ctx.get("properties", [])
                eq_classes = ctx.get("equivalent_classes", [])
                
                cluster_block += f"  Class: {cls_name}\n"
                if eq_classes:
                    cluster_block += f"  Equivalent Classes: {', '.join(eq_classes)}\n"
                cluster_block += f"  Class Description: {cls_desc}\n"
                #if props:
                    #cluster_block += f"  Properties: {', '.join(props)}\n"
            cluster_block += "\n"
        user_msg = (
            f"### Fuzzy Cluster of Entities\n{cluster_block}\n"
            "### Goal\nAnalyze these similar entity names to determine which ones are actual synonyms/typos "
            "of the exact same real-world concept, and which are distinct concepts that just happen to look similar. "
            "Return a rigorous resolution map where you assign each name an action of either 'equivalent' (if it points to a canonical target name) "
            "or 'keep_distinct' (if it represents an independent concept).\n"
            "If 'equivalent', provide the target_id it is equivalent to."
        )

        # JSON schema for OpenVINO GenAI structured generation
        resolution_schema = {
            "type": "object",
            "properties": {
                "resolved_instances": {
                    "type": "object",
                    "properties": {
                        name: {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["equivalent", "keep_distinct"]},
                                "target_id": {"type": ["string", "null"]}
                            },
                            "required": ["action", "target_id"],
                            "additionalProperties": False
                        } for name in cluster
                    },
                    "required": cluster,
                    "additionalProperties": False
                }
            },
            "required": ["resolved_instances"],
            "additionalProperties": False
        }

        # Query the NPU Backend
        raw_extractions, prompt_used = self.generate_with_schema(user_msg, resolution_schema)
        
        # Parse the outputs safely
        resolutions = []
        if raw_extractions and "resolved_instances" in raw_extractions:
            for name in cluster:
                # Map to target, fallback to 'keep_distinct' if omitted
                res = raw_extractions["resolved_instances"].get(name, {})
                action = res.get("action", "keep_distinct")
                target_id = res.get("target_id", None)
                
                # Safety check: ensure the mapped name is actually in the cluster
                if action == "equivalent" and target_id not in cluster:
                    target_id = None
                    action = "keep_distinct"
                    
                resolutions.append({
                    "original_id": name,
                    "action": action,
                    "target_id": target_id
                })
        else:
            # Fallback: flag all as 'keep_distinct'
            for name in cluster:
                resolutions.append({
                    "original_id": name,
                    "action": "keep_distinct",
                    "target_id": None
                })
            
        return resolutions, prompt_used
