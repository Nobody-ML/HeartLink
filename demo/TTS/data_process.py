import os

path = '/root/GPT-SoVITS/Hutao/'
files = os.listdir(path)


for id,value in enumerate(files):
    format = ' |hutao|zh| '.split('|')
    words = value.split('.')[0]
    format[3] = words
    format[0] = f'{path+str(id)}.wav'
    os.system(f'mv {path+value} {path+str(id)}.wav')
    list_content = '|'.join(format) + '\n'

    with open('/root/GPT-SoVITS/hutao.list','a') as f:
        f.write(list_content)