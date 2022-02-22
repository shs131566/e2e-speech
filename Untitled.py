import os
import json
import re

PATH = '/home/CORPUS/aihub_diquest/Training/[라벨]1.AI챗봇'
w = open('untitiled.txt', 'w')
hangul = re.compile('[^ 가-힣+]')

for dir_name in os.listdir(PATH):
    if os.path.isdir(os.path.join(PATH, dir_name)):
        for file in os.listdir(os.path.join(PATH, dir_name)):
            file_path = os.path.join(PATH, dir_name, file)

            with open(file_path, 'r') as f:
                json_data = json.load(f)
                if json_data['대화정보']['cityCode'] == '수도권':
                    text = json_data['발화정보']['stt']
                    if len(hangul.findall(text)) == 0:
                        wav = json_data['발화정보']['fileNm']
                        w.write(os.path.join(PATH, dir_name, wav) + '\t' + \
                            json_data['발화정보']['recrdTime'] + '\t' + text + '\n')
            
w.close() 