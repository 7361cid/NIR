import math
import nltk
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from readability import getmeasures

nltk.download('stopwords')

def frequancy_analis(words_list, key_word, max_new_words):  # Поиск новых ключевых слов, со схожей частотой
    new_key_words = []
    frequancy_dict = {}
    for unique_word in set(words_list):  # формирования словаря для подсчета частоты слов (ключи уникальны)
        frequancy_dict[unique_word] = 0

    for word in words_list:  # подсчет частот слов
        frequancy_dict[word] += 1

    if key_word in words_list:
        key_word_frequancy = frequancy_dict[key_word]
    else:
        key_word_frequancy = 0
    for word in words_list:
        #print("key word freq " + str(key_word_frequancy) + " word freq " + str(frequancy_dict[word]))
        if 0.8 * key_word_frequancy < frequancy_dict[word] < 1.1 * key_word_frequancy:
            new_key_words.append(word)

    new_key_words = list(set(new_key_words))
    if len(new_key_words) > max_new_words:
        return new_key_words[:max_new_words]   # Если найдено слишком много новых ключевых слов
    else:
        return new_key_words

def delete_stopwords(words_list):
    stopwords_ru = list(stopwords.words('russian'))
    words_list_without_stopwords = [word for word in words_list if word not in stopwords_ru]
    return words_list_without_stopwords


def lemmotize(word):
    morph = MorphAnalyzer()
    return morph.normal_forms(word)[0]


def N_gramma(*, word_list, token_length):
    new_word_list = []
    for i in range(len(word_list) - token_length + 1):
        token = ""
        for j in range(token_length):
            token += word_list[i+j] + " "
        new_word_list.append(token)
    return new_word_list


class TextKey:
    def __init__(self, key_info, ENG=False):
        self.key_info = key_info  # Текст ключевой последовательности
        self.key_words_list = self.get_key_words_list(ENG)  # Список слов
        print("self.key_info" + str(self.key_info))
        print("self.key_words_list" + str(self.key_words_list))

    @staticmethod
    def is_word(word):   # проверка на то что слово состоит только из букв
        for letter in word:
            if not letter.isalpha():
                return False
        return True

    def get_key_words_list(self, ENG):
        delete_bad_symbols(self.key_info)
        words_list = self.key_info.split()  # Разбиение по пробелам
        print("words_list " + str(words_list))
        if ENG:        # Для английского нужна другая лемматизация
            return words_list
        key_words_list_before_lemmatize = [word for word in words_list if self.is_word(word=word)]
        # проверка на то что слово только из букв
        key_words_list = [lemmotize(word.lower()) for word in key_words_list_before_lemmatize]
        print("key_words_list_before_lemmatize " + str(key_words_list_before_lemmatize))
        return delete_stopwords(key_words_list)


class TextDataBase:
    def __init__(self,  texts_list):
        self.texts_list = texts_list # список текстов
        self.doc_number = len(texts_list)  # количество документов
        self.doc_list = self.fill_data_base() # список элементы которого это списки слов документа
                                             # например 1ый элемент это список слов 1го документа

    def fill_data_base(self):
        data = []
        for text in self.texts_list:
            delete_bad_symbols(text)
            doc_words_list_before_lemmatize = text.split()
            doc_words_list = [lemmotize(word.lower()) for word in doc_words_list_before_lemmatize]
            data.append(delete_stopwords(doc_words_list))

        return data


class Finder:
    def __init__(self, text_key: TextKey, text_data_base: TextDataBase):
        self.text_key = text_key  # ключевая инормация
        self.text_data_base = text_data_base  # база документов
        self.statistic_list = self.make_TF_IDF_statistic() # статистика в виде списка, где каждый элемент это словарь
                                                           # ключевые словая в словаре это слова ключи, а значения
                                                           # это TF_IDF
    def TF_IDF(self, word, doc):
        #print("TF/IDF = " + str(self.TF(word, doc)) + "/" + str(self.IDF(word)) + " Для слова " + str(word) + " " + str(doc))
        return self.TF(word, doc) * self.IDF(word)

    def TF(self, word, doc):
        word_count = 0
        for doc_word in doc:
            if doc_word == word:
                word_count += 1
        return word_count

    def IDF(self, word):
        docs_with_word = self.number_docs_with_word(word)
        if docs_with_word != 0:
            return math.log10(self.text_data_base.doc_number / docs_with_word)  # Причина ошибки, если слово есть во всех текстах, то считается логарифм от log 1 = 0
        else:
            return 0  # Если слова нет ни в одном тексте?

    def number_docs_with_word(self, word):  # рассчет количества документов в которых есть слово, переданное функции
        docs_count = 0
        for doc_index in range(self.text_data_base.doc_number):
            if word in self.text_data_base.doc_list[doc_index]:
                docs_count += 1
        return docs_count

    def make_TF_IDF_statistic(self):
        statistic = []
        for doc_index in range(self.text_data_base.doc_number):
            dict_for_TF_IDF_for_doc = {}
            for word in self.text_key.key_words_list:
                print("self.text_key.key_words_list " + str(self.text_key.key_words_list))
                dict_for_TF_IDF_for_doc[word] = self.TF_IDF(word=word, doc=self.text_data_base.doc_list[doc_index])
            statistic.append(dict_for_TF_IDF_for_doc)
            print("dict_for_TF_IDF_for_doc " + str(dict_for_TF_IDF_for_doc))
        return statistic

    def search(self, tf_idf_avg):  # поиск, рассчет для каждого докмента суммы TF-IDF
        """
        tf_idf_avg - если True то tf-idf усредняется
        """
        rezult = []
        sum_TF_IDF_for_docs = []
        TF_IDF_vectors_for_docs = []
        doc_index = 0  # Индекс для поиска документа (нужно чтобы найти длину текста для усреднения)
        for doc_statistic in self.statistic_list:
            sum_TF_IDF = 0
            TF_IDF_vector = []
            print("doc_statistic.keys() " + str(doc_statistic.keys()))
            for key_word in doc_statistic.keys():
                sum_TF_IDF += doc_statistic[key_word]
                TF_IDF_vector.append(doc_statistic[key_word])
            TF_IDF_vectors_for_docs.append(TF_IDF_vector)

           # print("text_data_base.texts_list " +  str(self.text_data_base.texts_list))
           # print("text_data_base.doc_list" + str(self.text_data_base.doc_list))
            if tf_idf_avg and len(self.text_data_base.doc_list[doc_index]) != 0:
               # print("index " + str(doc_index) + " len= " + str(len(self.text_data_base.doc_list[doc_index])))
                sum_TF_IDF_for_docs.append(sum_TF_IDF/len(doc_statistic.keys()))
            else:
                sum_TF_IDF_for_docs.append(sum_TF_IDF)

            doc_index += 1

        for doc_index in range(self.text_data_base.doc_number):
            rezult_dict = {}
           # print("В документе " + str(doc_index)
           #        + " TF_IDF="+str(sum_TF_IDF_for_docs[doc_index]))
            rezult_dict["id"] = doc_index                           # 3аполняем словарь с результатами
            rezult_dict["TF-IDF"] = sum_TF_IDF_for_docs[doc_index]
            rezult_dict["text_words_list"] = self.text_data_base.doc_list[doc_index]
            rezult_dict["TF-IDF_Vector"] = TF_IDF_vectors_for_docs[doc_index]
            rezult_dict["ARI"] = ari(text=self.text_data_base.texts_list[doc_index])
            rezult.append(rezult_dict)
        return rezult


def search(key_info,  texts_list, tf_idf_avg, ENG=False):
    text_key = TextKey(key_info=key_info, ENG=ENG)
    text_data_base = TextDataBase(texts_list=texts_list)
    finder = Finder(text_key=text_key, text_data_base=text_data_base)
    return finder.search(tf_idf_avg=tf_idf_avg)


def delete_bad_symbols(text):
    for bad_symbol in [',', '.', '!', '?', '(', ')']:
        text = text.replace(bad_symbol, ' ')


def ari(*, text):
    measures = getmeasures(text=text, lang="en")  # Для ari язык не важен
    return measures['readability grades']['ARI']


if __name__ == "__main__":
    key_info = "после вечера у Ростовых Графиня была"
    texts_list = ["после вечера", "у Ростовых ", "Графиня была"]
    search(key_info=key_info, texts_list=texts_list, tf_idf_avg=True)
    ari(text=key_info)

    
