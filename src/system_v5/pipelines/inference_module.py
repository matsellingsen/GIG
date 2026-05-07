import sys
sys.path.append("C:\\Users\\matse\\gig\\src\\system_v5") # Adjust this path to point to the root of your project if needed

# import setup function
from backends import load_backend
from tools.ttl_handling.load_ttl import load_ttl
from tools.ttl_handling.resolve_ttl_path import resolve_ttl_path

# import functions for each step of the pipeline
from tools.inference_module.input_to_graph import atomic_to_graph
from tools.inference_module.fetch_relevant_info import fetch_relevant_info
from tools.inference_module.generate_answer import generate_answer
from tools.inference_module.map_answer_to_context import map_answer_to_context
from tools.inference_module.validate_answer import validate_answer

# Import agents used in the pipeline
from agent_loop.agents.inference_module.extract_question_type_agent import ExtractQuestionTypeAgent
from agent_loop.agents.inference_module.resolve_answer_form_agent import ResolveAnswerFormAgent
from agent_loop.agents.inference_module.extract_entity_agent import ExtractEntityAgent
from agent_loop.agents.inference_module.extract_relation_agent import ExtractRelationAgent
from agent_loop.agents.inference_module.extract_object_agent import ExtractObjectAgent
from agent_loop.agents.inference_module.resolve_entity_agent import ResolveEntityAgent
from agent_loop.agents.inference_module.generate_answer_agent import GenerateAnswerAgent
from agent_loop.agents.inference_module.validate_answer_agent import ValidateAnswerAgent    

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
        self.validate_answer_agent = ValidateAnswerAgent(backend=self.backend)

        # load ttl for fetching relevant info (this can be moved to a different part of the code if needed, but for simplicity we load it here)
        resolved_ttl_path = resolve_ttl_path(ttl_path=None)
        self.ttl = load_ttl(file_path=resolved_ttl_path)

    def run(self) -> dict:
        # get input
        atomic_input = input("Ask an atomic question: ")

        # Step 1: Convert the input question in ATOMIC format into a structured graph representation (triplet + question type)
        question_info = atomic_to_graph(atomic_input=atomic_input, 
                                        extract_question_type_agent=self.extract_question_type_agent, 
                                        extract_answer_form_agent=self.extract_answer_form_agent,
                                        extract_entity_agent=self.extract_entity_agent, 
                                        extract_relation_agent=self.extract_relation_agent, 
                                        extract_object_agent=self.extract_object_agent)


        # Step 2: Fetch relevant information from the knowledge graph based on the extracted triplet and question type
        fetched_relevant_info = fetch_relevant_info(question_info=question_info,
                                            ttl=self.ttl,
                                            resolve_entity_agent=self.resolve_entity_agent)
        

        if fetched_relevant_info == "noPrimaryEntityFound":
            print(f"Could not find an entity in the KB that fits for the extracted entity {question_info.get('entity').get('value')}. Cannot proceed with answer generation.")
            print("Please try again with a different question.")
            self.run() # restart the process for a new question
            return            
        
        entity_context = fetched_relevant_info.get("entity_context") #unpack the relevant info from the full context
        object_context = fetched_relevant_info.get("object_context") #get the object context

        # Step 3: Generate an answer to the question based on the retrieved relevant information
        answer = generate_answer(question_info=question_info,
                                 entity_context=entity_context,
                                 object_context=object_context,
                                 generate_answer_agent=self.generate_answer_agent)
        
        # Step 4: Map the generated answer back to the ontology context
        answer_text = answer.get("answer")
        reasoning_text = answer.get("reasoning")
        print("================================")
        print(f"Generated answer before mapping: {answer_text}")
        print(f"Generated reasoning before mapping: {reasoning_text}")
        #print(f"Flattened relevant info used to show the agent: {relevant_info_flat}")
        print(f"entity info used for mapping: {entity_context}")
        print("================================")
        mapped_entity_answer = map_answer_to_context(answer=answer_text, context=entity_context)
        mapped_reasoning_entity_answer = map_answer_to_context(answer=reasoning_text, context=entity_context)
        mapped_object_answer = map_answer_to_context(answer=answer_text, context=object_context) if object_context is not None else None
        print(f"Mapped entity answer: {mapped_entity_answer}")
        print(f"Mapped reasoning answer: {mapped_reasoning_entity_answer}")
        if object_context is not None:
            print(f"Mapped object answer: {mapped_object_answer}")

        # Step 5: Validate the generated answer based on the question, the extracted triplet, and the retrieved relevant information
        #validation_result = validate_answer(answer=answer,
        #                                  question_info=question_info,
        #                                  entity_context=entity_context,
        #                                  object_context=object_context,
        #                                  mapped_entity_answer=mapped_entity_answer,
        #                                  mapped_object_answer=mapped_object_answer,
        #                                  validate_answer_agent=self.validate_answer_agent)
        #if validation_result:
        #    print(f"Final answer: {answer_text}")
        #else:
        #    print("The generated answer did not pass final validation.")

def main():
    inference_module = InferenceModule()
    inference_module.run()

if __name__ == "__main__":
    main()