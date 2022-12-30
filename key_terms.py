import string
from lxml import etree
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

header_list = []


def tokenize(text):
    # tokenize text
    tokens = word_tokenize(text.lower())
    return tokens


def lemmatize(tokens):
    lemmatize = WordNetLemmatizer()
    lemmatized_tokens = [lemmatize.lemmatize(word) for word in tokens if word != "has" and word != "was" and word != "us" and word != "as"]
    return lemmatized_tokens


def remove_stopword_punctuation(lemmatized_tokens):
    stopwords_list = list(stopwords.words('english')) + ['ha', 'wa', 'u', 'a']
    punctuation_list = list(string.punctuation)
    # get rid of stopwords and punctuation
    tokens_without_stopwords_punctuation = [word for word in lemmatized_tokens \
                                            if word not in stopwords_list \
                                            and word not in punctuation_list]
    return tokens_without_stopwords_punctuation


def extract_noun(tokens_without_stopwords_punctuation):
    tokens_tag_noun = [token for token in tokens_without_stopwords_punctuation \
                       if nltk.pos_tag([token])[0][1] == 'NN']
    return tokens_tag_noun


def count_tfidf_sort_pick_best(tokens_tag_noun, AMOUNT):
    dataset = []
    for list in tokens_tag_noun:
        dataset.append(" ".join(tokens_tag_noun))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset)
    tfidf_matrix = tfidf_matrix.toarray()
    dimension = tfidf_matrix.shape
    terms = vectorizer.get_feature_names_out()
    row = 0
    while row < dimension[0]:
        # print(f"row: {row}, dimension[0]: {dimension[0]}")
        column = 0
        word_rate = []
        while column < dimension[1]:
            # print(f"column: {column}, dimension[1]: {dimension[1]}")
            if tfidf_matrix[row][column] != 0:
                word_rate.append((terms[column], tfidf_matrix[row][column]))  # first value = word , second value = rate
            column += 1
        best_scoring_words = sorted(word_rate, key=lambda x: (x[1], x[0]), reverse=True)[0:AMOUNT]
        row += 1
    return best_scoring_words


# def print_best_scoring_words(sorted_list, amount):
#     # sort token_list with descending order
#     freq_list = Counter(sorted_list)
#     # print(freq_list)
#     most_common_freq = freq_list.most_common()
#     for i in range(amount):
#         print(most_common_freq[i][0], end=" ")
#     print()


def main():
    global header_list
    xml_file = "news.xml"
    root = etree.parse(xml_file).getroot()
    AMOUNT = 5
    # etree.dump(root)
    for corpus in root:
        for news in corpus:
            for value in news:
                if value.get('name') == "head":
                    header = value.text
                    header_list.append(header)
    for corpus in root:
        news_index = 0
        for news in corpus:
            for value in news:
                if value.get('name') == "text":
                    text = value.text
                    tokens = tokenize(text)
                    lemmatized_tokens = lemmatize(tokens)
                    tokens_without_stopwords_punctuation = remove_stopword_punctuation(lemmatized_tokens)
                    tokens_tag_noun = extract_noun(tokens_without_stopwords_punctuation)
                    best_scoring_words = count_tfidf_sort_pick_best(tokens_tag_noun, AMOUNT)
                    print(f"{header_list[news_index]}:")
                    for tuple in best_scoring_words:
                        print(tuple[0], end=' ')
                    print()
            news_index += 1



if __name__ == "__main__":
    main()
