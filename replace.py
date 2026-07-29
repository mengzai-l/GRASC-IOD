import os
folder_path = './'
new_string = 'your_path_to_VOC2007'
old_string = '/xxx/yyy/zzz/VOC2007'

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.py') and filename != 'replace.py':
            file_path = os.path.join(root, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            new_content = content.replace(old_string, new_string)
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)
