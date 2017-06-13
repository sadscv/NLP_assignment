import requests

url_prefix = "http://222.204."


def find():
    for i in range(2,255):
        for j in range(1,255):
            print("正在查找"+url_prefix + str(i) +"."+ str(j))
            url = url_prefix + str(i) +"."+ str(j)+":34531"
            try:
                if(requests.post(url,timeout=0.01).content == b'echoissb'):
                    print('found')
                    result = url
                    break
                    print(url.split(":")[1])
                    return
            except:
                pass
    print(result)

if __name__ == '__main__':
    find()