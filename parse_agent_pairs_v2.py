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
                
                # Extract system prompt path (multi-line friendly)
                sys_prompt_path_match = re.search(r'load_prompt\(\s*(?:r?["\'])(.*?\.txt)["\']\s*\)', content)
                
                # Check for dynamic ones if there are multiple load_prompt usages
                sys_prompts = []
                for match in re.finditer(r'load_prompt\(\s*(?:r?["\'])(.*?\.txt)["\']\s*\)', content):
                    prompt_path = match.group(1).replace('\\\\', '\\')
                    prompt_path = os.path.abspath(prompt_path)
                    try:
                        with open(prompt_path, 'r', encoding='utf-8') as sp_file:
                            sys_prompts.append(sp_file.read())
                    except Exception as e:
                        sys_prompts.append(f"Could not read system prompt at {prompt_path}: {e}")
                
                sys_prompt_content = "\n\n--- ANOTHER SYSTEM PROMPT FOUND ---\n\n".join(sys_prompts) if sys_prompts else "SYSTEM PROMPT NOT FOUND"
                
                # Extract user message construction
                user_msg_code = []
                lines = content.split('\n')
                capture = False
                for i, line in enumerate(lines):
                    if (re.search(r'^\s*(user_msg|user_message|user_prompt|prompt)\s*=', line)):
                        capture = True
                        user_msg_code.append(line.strip())
                        # Check multiline
                        if line.count('(') > line.count(')') or line.count('"""') % 2 != 0 or line.count("'''") % 2 != 0:
                            parens = line.count('(') - line.count(')')
                            triple_quotes = line.count('"""') % 2 != 0
                            j = i + 1
                            while j < len(lines) and (parens > 0 or triple_quotes):
                                user_msg_code.append(lines[j].strip())
                                parens += lines[j].count('(') - lines[j].count(')')
                                if '"""' in lines[j]:
                                    triple_quotes = not triple_quotes
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
