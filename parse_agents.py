import os
import ast

agents_dir = 'c:/Users/matse/gig/src/system_v5/agent_loop/agents'
prompts_dir = 'c:/Users/matse/gig/src/system_v5/prompts/system'

with open('agent_prompts.txt', 'w', encoding='utf-8') as out:
    for root, _, files in os.walk(agents_dir):
        for f in files:
            if f.endswith('.py') and f not in ('__init__.py', 'agent.py', 'base_ontology_agent.py'):
                path = os.path.join(root, f)
                with open(path, 'r', encoding='utf-8') as p:
                    content = p.read()
                    
                # We need to find the user message and system prompt
                # The user message might be constructed in execute() or defined as a template.
                out.write(f"=== {f} ===\n")
                out.write("SYSTEM PROMPT PATHS / CONTENT:\n")
                for line in content.split('\n'):
                    if 'load_prompt' in line and '.txt' in line:
                        out.write(line.strip() + '\n')
                out.write("USER MESSAGE Construction:\n")
                
                # Simple extraction: look for strings that contain '{' or look like f-strings or prompt templates
                for i, line in enumerate(content.split('\n')):
                    if 'prompt =' in line or 'user_message =' in line or 'user_prompt =' in line or 'message =' in line:
                        out.write(line.strip() + '\n')
                out.write("\n\n")

print("Done")
