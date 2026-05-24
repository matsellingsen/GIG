import os
import re

agents_dir = 'c:/Users/matse/gig/src/system_v5/agent_loop/agents'
prompts_dir = 'c:/Users/matse/gig/src/system_v5/prompts/system'

with open('agent_prompts_pairs.txt', 'w', encoding='utf-8') as out:
    for root, _, files in os.walk(agents_dir):
        for f in files:
            if f.endswith('.py') and f not in ('__init__.py', 'agent.py', 'base_ontology_agent.py'):
                path = os.path.join(root, f)
                with open(path, 'r', encoding='utf-8') as p:
                    content = p.read()
                
                # Extract system prompt path
                sys_prompt_path_match = re.search(r'load_prompt\((?:r?["\'])(.*?\.txt)["\']\)', content)
                sys_prompt_content = "SYSTEM PROMPT NOT FOUND"
                if sys_prompt_path_match:
                    prompt_path = sys_prompt_path_match.group(1).replace('\\\\', '\\')
                    # normalize
                    prompt_path = os.path.abspath(prompt_path)
                    try:
                        with open(prompt_path, 'r', encoding='utf-8') as sp_file:
                            sys_prompt_content = sp_file.read()
                    except Exception as e:
                        sys_prompt_content = f"Could not read system prompt at {prompt_path}: {e}"
                
                # Extract user message construction
                user_msg_code = []
                lines = content.split('\n')
                capture = False
                for i, line in enumerate(lines):
                    if (re.search(r'^\s*(user_msg|user_message|user_prompt|prompt)\s*=', line)):
                        capture = True
                        user_msg_code.append(line.strip())
                        # If it's a multiline string/parentheses, capture until closed
                        if line.count('(') > line.count(')'):
                            parens = line.count('(') - line.count(')')
                            j = i + 1
                            while parens > 0 and j < len(lines):
                                user_msg_code.append(lines[j].strip())
                                parens += lines[j].count('(') - lines[j].count(')')
                                j += 1
                        capture = False

                
                out.write(f"=== {f} ===\n")
                out.write("--- SYSTEM PROMPT ---\n")
                out.write(sys_prompt_content.strip() + "\n")
                out.write("--- USER MESSAGE (Code Construction) ---\n")
                if user_msg_code:
                    out.write("\n".join(user_msg_code) + "\n")
                else:
                    out.write("No user message pattern found.\n")
                out.write("\n\n")

print("Done")
