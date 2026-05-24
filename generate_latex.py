import re

with open('agent_prompts_pairs.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Split by the separator for new agents
blocks = text.split('=== ')

with open('agent_prompts_latex.tex', 'w', encoding='utf-8') as out:
    for block in blocks:
        if not block.strip():
            continue
            
        # Parse agent name and content
        lines = block.split('\n')
        agent_filename = lines[0].replace(' ===', '').strip()
        agent_name = agent_filename.replace('_agent.py', '').replace('.py', '').replace('_', ' ').title()
        
        # Write subsection
        out.write(f"\\subsubsection*{{{agent_name}}}\n\n")
        
        # Find System Prompt
        sys_start = block.find('--- SYSTEM PROMPT ---') + len('--- SYSTEM PROMPT ---\n')
        sys_end = block.find('--- USER MESSAGE (Code Construction) ---')
        
        if sys_start != -1 and sys_end != -1:
            sys_prompt = block[sys_start:sys_end].strip()
            
            # Handle the "ANOTHER SYSTEM PROMPT FOUND" split for extract_object
            if '--- ANOTHER SYSTEM PROMPT FOUND ---' in sys_prompt:
                sub_prompts = sys_prompt.split('--- ANOTHER SYSTEM PROMPT FOUND ---')
                out.write(f"\\paragraph*{{System Prompt (Standard)}}\n")
                out.write("\\begin{verbatim}\n")
                out.write(sub_prompts[0].strip() + "\n")
                out.write("\\end{verbatim}\n\n")
                
                out.write(f"\\paragraph*{{System Prompt (Comparative)}}\n")
                out.write("\\begin{verbatim}\n")
                out.write(sub_prompts[1].strip() + "\n")
                out.write("\\end{verbatim}\n\n")
            else:
                out.write(f"\\paragraph*{{System Prompt}}\n")
                out.write("\\begin{verbatim}\n")
                out.write(sys_prompt + "\n")
                out.write("\\end{verbatim}\n\n")
                
        # Find User Message
        usr_start = sys_end + len('--- USER MESSAGE (Code Construction) ---\n')
        if sys_end != -1:
            usr_msg = block[usr_start:].strip()
            
            # Strip the f""" and """ wrappers if present to make it cleaner
            if usr_msg.startswith('user_msg = f"""'):
                usr_msg = usr_msg[len('user_msg = f"""'):].strip()
            elif usr_msg.startswith('user_msg = (\nf"'):
                usr_msg = usr_msg[len('user_msg = (\nf"'):].strip()
            
            if usr_msg.endswith('"""'):
                usr_msg = usr_msg[:-3].strip()
            elif usr_msg.endswith('")'):
                usr_msg = usr_msg[:-2].strip()
                
            out.write(f"\\paragraph*{{User Prompt Template}}\n")
            out.write("\\begin{verbatim}\n")
            out.write(usr_msg + "\n")
            out.write("\\end{verbatim}\n\n")
            
        out.write("\\vspace{1em}\n\n")

print("Done generating LaTeX")
