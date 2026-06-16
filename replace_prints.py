import os
import re

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add imports if not present
    if 'import logging' not in content:
        content = re.sub(r'^(import [^\n]+)', r'\1\nimport logging', content, count=1, flags=re.MULTILINE)
        content = re.sub(r'(import logging\n)', r'\1\nlogger = logging.getLogger(__name__)\n', content, count=1)
    
    # Replace print statements
    def replacer(match):
        indent = match.group(1)
        inner = match.group(2)
        if '[WARNING]' in inner:
            return f'{indent}logger.warning({inner})'
        elif '[ERROR]' in inner:
            return f'{indent}logger.error({inner})'
        else:
            return f'{indent}logger.info({inner})'

    new_content = re.sub(r'^(\s*)print\((.*)\)$', replacer, content, flags=re.MULTILINE)
    
    if content != new_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")
    else:
        print(f"No changes in {filepath}")

base_dir = r"d:\Programming\hackaton\AgenticTradingBot-Team-Vegeta\trading_bot"
process_file(os.path.join(base_dir, "nodes.py"))
process_file(os.path.join(base_dir, "graph.py"))
