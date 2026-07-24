import base64
import os
import re

# Read the updated presentation.html
with open('presentation.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Find all image src references (relative paths ../data/plots/...)
img_pattern = re.compile(r'src="(\.\./data/plots/[^"]+)"')
matches = list(set(img_pattern.findall(html)))

print('Found images:')
for m in matches:
    print(' ', m)
print('Total unique:', len(matches))

# Convert each image to base64
replaced = 0
for rel_path in matches:
    # Build absolute path from presentation/ folder
    abs_path = os.path.join(os.path.dirname(os.path.abspath('presentation.html')), rel_path)
    abs_path = os.path.normpath(abs_path)
    
    if not os.path.exists(abs_path):
        print(f'  WARNING: File not found: {abs_path}')
        continue
    
    # Determine mime type
    ext = os.path.splitext(abs_path)[1].lower()
    mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.svg': 'image/svg+xml'}
    mime = mime_map.get(ext, 'image/png')
    
    with open(abs_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    
    data_uri = f'data:{mime};base64,{b64}'
    # Replace all occurrences of this src
    html = html.replace(f'src="{rel_path}"', f'src="{data_uri}"')
    replaced += 1
    print(f'  Embedded: {rel_path} ({os.path.getsize(abs_path):,} bytes)')

print(f'\nEmbedded {replaced} images.')

# Write standalone file
with open('presentation_standalone.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('Done! Written to presentation_standalone.html')
print(f'File size: {os.path.getsize("presentation_standalone.html"):,} bytes')
