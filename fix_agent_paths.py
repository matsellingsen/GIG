import os
import re

agents_dir = 'c:/Users/matse/gig/src/system_v5/agent_loop/agents'
prompts_dir = 'c:/Users/matse/gig/src/system_v5/prompts/system'

# Build mapping of prompt txt filename to its correct relative path (relative to repo root or absolute)
prompt_map = {}
for root, _, files in os.walk(prompts_dir):
    for f in files:
        if f.endswith('.txt'):
            # Convert to forward slashes for safer regex replacement
            prompt_map[f] = os.path.abspath(os.path.join(root, f)).replace('\\', '/')

fixed_count = 0
for root, _, files in os.walk(agents_dir):
    for f in files:
        if f.endswith('.py') and f not in ('__init__.py', 'agent.py', 'base_ontology_agent.py'):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8') as p:
                content = p.read()
            
            # Find all load_prompt occurrences
            new_content = content
            changed = False
            matches = list(re.finditer(r'load_prompt\s*\(\s*(r?f?["\'])(.*?\.txt)(["\'])\s*\)', content, re.DOTALL))
            for match in matches:
                full_call = match.group(0)
                quote_start = match.group(1)
                inner_path = match.group(2)
                quote_end = match.group(3)
                
                filename = os.path.basename(inner_path.replace('\\', '/'))
                actual_path = prompt_map.get(filename)
                
                if actual_path:
                    # Only repair if the text inside the python code doesn't exist
                    # Let's see if evaluating it directly fails
                    test_path = inner_path.replace('\\\\', '\\')
                    if not os.path.exists(test_path):
                        # Construct a clean raw string format to write back
                        new_inner_path = actual_path.replace('/', '\\\\')
                        new_call = f'load_prompt({quote_start}{new_inner_path}{quote_end})'
                        new_content = new_content.replace(full_call, new_call)
                        changed = True

            if changed:
                with open(path, 'w', encoding='utf-8') as p:
                    p.write(new_content)
                fixed_count += 1
                print(f"Fixed broken prompt paths in {f}")

print(f"Done. Fixed {fixed_count} files.")
