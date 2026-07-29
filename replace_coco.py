import os

folder_path = './'

new_string = 'your_path_to_COCO2017'

old_string = '/xxx/yyy/zzz/COCO/2017'

count = 0
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.py') and filename not in ['replace.py', 'replace_coco.py']:
            file_path = os.path.join(root, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # replace：这样 /xxx/yyy/zzz/COCO/2017/train2017 就会变成 /home/.../data/COCO/train2017
            new_content = content.replace(old_string, new_string)

            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)
                count += 1

