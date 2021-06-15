import re

def extract(text,):
    pat = re.compile(" - val_loss: (\d+(\.\d+)?)")
    matches = pat.findall(text)
    if matches:
        for m in matches:
            print(m[0],end=',')
    print("")

if __name__ == '__main__':
    with open("record.txt", 'r') as f:
        text = f.read()
    extract(text)
