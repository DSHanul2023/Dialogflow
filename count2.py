import csv
import chardet
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

def extract_most_common_nouns(csv_file):
    word_groups = {}

    encoding = detect_encoding(csv_file)
    with open(csv_file, 'r', encoding=encoding) as file:
        reader = csv.reader(file)
        for row in reader:
            first_word = row[0]  # 첫 번째 열의 단어 추출
            words = ' '.join(row[1:]).split()  # 두 번째 열 이후의 단어를 공백 기준으로 분리하여 추출
            nouns = extract_nouns(words)  # 명사 추출
            if first_word not in word_groups:
                word_groups[first_word] = []
            word_groups[first_word].extend(nouns)

    most_common_nouns_by_first_word = {}
    for first_word, nouns in word_groups.items():
        noun_counts = Counter(nouns)
        most_common_nouns = noun_counts.most_common(10)  # 가장 많이 사용된 4가지 명사 추출
        filtered_nouns = remove_stopwords(most_common_nouns)  # 불용어 제외
        most_common_nouns_by_first_word[first_word] = filtered_nouns

    return most_common_nouns_by_first_word

def extract_nouns(words):
    tagged_words = pos_tag(words)  # 품사 태깅
    nouns = [word for word, pos in tagged_words if pos.startswith('NN')]  # 명사 필터링
    return nouns

def remove_stopwords(nouns):
    stopwords = [
        '이', '그', '저', '것', '수', '등', '들', '우리', '당신', '자신', '너', '그들', '무엇', '어떤', '누구',
        '나', '너희', '우리들', '그들', '여러분', '안', '못', '개', '에서', '으로', '하다', '하는', '된', '된다', '하고', '한다', '이다',
        '그리고', '그러나', '하지만', '그런데', '그래서', '또는', '때문에', '그럼', '그러므로', '그렇지만', '이렇게', '이렇다', '이러한', '다른', '아니라',
        '하는', '하는데', '될', '수도', '있다', '까지', '하면서', '그러면', '된다', '인', '위', '밑', '중', '쯤', '위해', '데', '로',
        '에', '가', '를', '을', '는', '은', '이', '가', '과', '와', '들', '들의', '을', '를', '에게', '에게서', '로', '에서', '거',
        '더', '줄', '좀', '번', '쪽', '좀', '게', '거', '너무', '다', '제가', '게', '내', '한', '잔', '건', '걸로', '정말'
    ]
    filtered_nouns = [(noun, count) for noun, count in nouns if noun not in stopwords]
    return filtered_nouns

# CSV 파일 경로 지정
csv_file_path = 'test2.csv'

# 함수 호출하여 첫 번째 열의 단어별 가장 많이 사용된 명사 추출
result = extract_most_common_nouns(csv_file_path)

print("첫 번째 열의 단어별 가장 많이 사용된 명사:")
for first_word, most_common_nouns in result.items():
    print(first_word, ":")
    for noun, count in most_common_nouns:
        print(noun, ":", count)
    print()