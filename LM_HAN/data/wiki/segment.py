import jieba
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    # load stopwords set
    stopwordset = set()
    with open('stopwords.txt','r',encoding='utf-8') as sw:
        for line in sw:
            stopwordset.add(line.strip('\n'))

    output = open('wiki_seg.txt','w')
    
    texts_num = 0
    
    with open('wiki_chs.txt','r') as content :
        for line in content:
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopwordset:
                    output.write(word +' ')
            texts_num += 1
            if texts_num % 10000 == 0:
                logging.info("已完成前 %d 行的斷詞" % texts_num)
    output.close()
    
if __name__ == '__main__':
	main()
