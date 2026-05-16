import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed
import os
import json
from datetime import datetime
import uuid

# import setup function
from backends import load_backend
from tools.ttl_handling.load_ttl import load_ttl
from tools.ttl_handling.resolve_ttl_path import resolve_ttl_path

# import functions for each step of the pipeline
from tools.inference_module.input_to_graph import atomic_to_graph
from tools.inference_module.fetch_relevant_info import fetch_relevant_info
from tools.inference_module.generate_answer import generate_answer
from tools.inference_module.map_answer_to_context import map_answer_to_context, merge_mappings

# Import agents used in the pipeline
from agent_loop.agents.inference_module.extract_question_type_agent import ExtractQuestionTypeAgent
from agent_loop.agents.inference_module.resolve_answer_form_agent import ResolveAnswerFormAgent
from agent_loop.agents.inference_module.extract_entity_agent import ExtractEntityAgent
from agent_loop.agents.inference_module.extract_relation_agent import ExtractRelationAgent
from agent_loop.agents.inference_module.extract_object_agent import ExtractObjectAgent
from agent_loop.agents.inference_module.resolve_entity_agent import ResolveEntityAgent
from agent_loop.agents.inference_module.generate_answer_agent import GenerateAnswerAgent

class InferenceModule:
    def __init__(self):
        
        # init backend
        self.backend = load_backend(name="phi-npu-openvino")

        #init agents used in the pipeline
        self.extract_question_type_agent = ExtractQuestionTypeAgent(backend=self.backend)
        self.extract_answer_form_agent = ResolveAnswerFormAgent(backend=self.backend)
        self.extract_entity_agent = ExtractEntityAgent(backend=self.backend)
        self.extract_relation_agent = ExtractRelationAgent(backend=self.backend)
        self.extract_object_agent = ExtractObjectAgent(backend=self.backend)
        self.resolve_entity_agent = ResolveEntityAgent(backend=self.backend)
        self.generate_answer_agent = GenerateAnswerAgent(backend=self.backend)

        # load ttl for fetching relevant info (this can be moved to a different part of the code if needed, but for simplicity we load it here)
        resolved_ttl_path = resolve_ttl_path(ttl_path=None)
        self.ttl = load_ttl(file_path=resolved_ttl_path)

    def run(self) -> dict:
        while True:

            # Init inference log for this question
            global inference_log
            inference_log = {}

            # get input
            atomic_input = input("Ask an atomic question: ")
            if atomic_input is None:
                return
            atomic_input = atomic_input.strip()
            inference_log["input"] = atomic_input
            if atomic_input == "":
                print("Empty input, exiting.")
                return

            # Step 1: Convert the input question in ATOMIC format into a structured graph representation (triplet + question type & answer form)
            question_info = atomic_to_graph(atomic_input=atomic_input, 
                                            extract_question_type_agent=self.extract_question_type_agent, 
                                            extract_answer_form_agent=self.extract_answer_form_agent,
                                            extract_entity_agent=self.extract_entity_agent, 
                                            extract_relation_agent=self.extract_relation_agent, 
                                            extract_object_agent=self.extract_object_agent,
                                            inference_log=inference_log)

            if isinstance(question_info, str): # Check if the output is an error message
                inference_log["error"] = question_info
                print(question_info)
                return
            
            # Step 2: Fetch relevant information from the knowledge graph based on the extracted triplet and question type
            fetched_relevant_info = fetch_relevant_info(question_info=question_info,
                                                ttl=self.ttl,
                                                resolve_entity_agent=self.resolve_entity_agent,
                                                inference_log=inference_log)


            if isinstance(fetched_relevant_info, str): # Check if the output is an error message
                inference_log["error"] = fetched_relevant_info
                print(fetched_relevant_info)
                return

            entity_context = fetched_relevant_info.get("entity_context") #unpack the relevant info from the full context
            object_context = fetched_relevant_info.get("object_context") #get the object context

            # Step 3: Generate an answer to the question based on the retrieved relevant information
            answer = generate_answer(question_info=question_info,
                                     entity_context=entity_context,
                                     object_context=object_context,
                                     generate_answer_agent=self.generate_answer_agent,
                                     inference_log=inference_log)
            
            if isinstance(answer, str): # Check if the output is an error message
                inference_log["error"] = answer
                print(answer)
                return
        
            # Step 4: Map the generated answer back to the ontology context
            answer_text = answer.get("answer")
            reasoning_text = answer.get("reasoning")
            print("================================")
            print(f"Generated reasoning: {reasoning_text}")
            print(f"Generated answer: {answer_text}")
            print(f"entity info used for mapping: {entity_context}")
            print("================================")

            mapped_reasoning_entity_answer = map_answer_to_context(answer=reasoning_text, context=entity_context, inference_log=inference_log)
            if isinstance(mapped_reasoning_entity_answer, str): # Check if the output is an error message
                inference_log["error"] = mapped_reasoning_entity_answer
                print(mapped_reasoning_entity_answer)
                return
            
            mapped_answer_entity_answer = map_answer_to_context(answer=answer_text, context=entity_context, inference_log=inference_log)
            if isinstance(mapped_answer_entity_answer, str): # Check if the output is an error message
                inference_log["error"] = mapped_answer_entity_answer
                print(mapped_answer_entity_answer)
                return
            
            mapped_entity_merged = merge_mappings(mapped_reasoning_entity_answer, mapped_answer_entity_answer)

            mapped_reasoning_object_answer = None
            mapped_answer_object_answer = None
            mapped_object_merged = None
            
            if object_context is not None:
                mapped_reasoning_object_answer = map_answer_to_context(answer=reasoning_text, context=object_context, inference_log=inference_log)
                if isinstance(mapped_reasoning_object_answer, str): # Check if the output is an error message
                    inference_log["error"] = mapped_reasoning_object_answer
                    print(mapped_reasoning_object_answer)
                    return
                
                mapped_answer_object_answer = map_answer_to_context(answer=answer_text, context=object_context, inference_log=inference_log)
                if isinstance(mapped_answer_object_answer, str): # Check if the output is an error message
                    inference_log["error"] = mapped_answer_object_answer
                    print(mapped_answer_object_answer)
                    return
                
                mapped_object_merged = merge_mappings(mapped_reasoning_object_answer, mapped_answer_object_answer)
           
            print(f"merged entity mapping: {mapped_entity_merged}")
            if object_context is not None:
                print(f"merged object mapping: {mapped_object_merged}")
            # add merged mappings to inference_log
            try:
                inference_log["mapped_entity_merged"] = mapped_entity_merged
                inference_log["mapped_object_merged"] = mapped_object_merged
            except Exception:
                pass

            # Persist a copy of the full inference_log into src/system_v5/inference_logs/
            try:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                logs_dir = os.path.join(base_dir, "inference_logs")
                os.makedirs(logs_dir, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
                unique = uuid.uuid4().hex[:8]
                fname = f"inference_{ts}_{unique}.json"
                full_path = os.path.join(logs_dir, fname)
                with open(full_path, "w", encoding="utf-8") as fh:
                    json.dump(inference_log, fh, ensure_ascii=False, indent=2)
                print(f"Saved mapped inference log to: {full_path}")
            except Exception as e:
                print(f"Warning: failed to write inference mapped log: {e}")
                

def main():
    inference_module = InferenceModule()
    inference_module.run()

if __name__ == "__main__":
    main()