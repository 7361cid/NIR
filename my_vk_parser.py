import requests
import text_search
import word_vector
import lemotizer


def get_text_from_vk(*, query, domain, count, lemotize=False, Ngram=0):
    """
    Теперь поиск работает по отдельным термам (по словам или N-граммам)
    """
    key_words_list = query.split()  # Разбиение по пробелам
    texts_from_vk = []
    token_auth = '61249f3161249f3161249f31aa615382936612461249f310145a604685c59e9ecda0d4b'   # токен авторизаци
    version = '5.130'
    if Ngram:
        text_search.N_gramma(word_list=key_words_list, token_length=Ngram)
    for key_word in key_words_list:  # поиск по отдельным термам
        if lemotize:
            key_word = lemotizer.lemmotize(key_word)
        response = requests.get('https://api.vk.com/method/wall.search',
                                params={
                                    "access_token": token_auth,
                                    "v": version,
                                    "count": count,
                                    "domain": domain,
                                    "query": key_word,
                                }
                                )
        data = response.json()["response"]["items"]
        texts_with_key_word = [d["text"] for d in data]  # тексты с отдельным термом
        texts_from_vk.extend(texts_with_key_word)   # добавление текстов с отдельным термом  к общему списку текстов
        for text in texts_from_vk:
            text_search.delete_bad_symbols(text)

    return texts_from_vk


def user_chouse_emulator(*, texts_list, random):  # Имитация выбора пользователя
    if random:
        return random.choice(texts_list)
    else:
        for text in texts_list:
            if len(text) > 0:
                return text


if __name__ == "__main__":
    key_info = "каждый раунд"  # взял текст из одного поста, чтобы точно было совпадение
    texts_list = get_text_from_vk(query=key_info, count=5, domain='sports_books')    # поиск постов с ключевыми словами

    user_text = user_chouse_emulator(texts_list)  # выбор текста пользователем (один случайный текст)
    for bad_symbol in [',', '.', '!', '?']:  # выкинуть из текста символы пунтуации, чтобы не было слов в стиле "Я,"
        key_info.replace(bad_symbol, '')
        user_text.replace(bad_symbol, '')

    key_list = key_info.split()  # Разбиение по пробелам
    words_list = user_text.split()
    new_key_list = []
    for key in key_list:  # поиск новых ключевых слов  (Если новых ключевых слов слишком много?)
        new_key_list.extend(text_search.frequancy_analis(words_list=words_list, key_word=key, max_new_words=2))
    new_texts_list = get_text_from_vk(query=key_info, count=5, domain='sports_books')    # поиск постов с расширенным
                                                                                         # набором ключевых слов

    rezult = text_search.search(key_info=key_info, texts_list=new_texts_list)  # расчет TF-IDF
    rezult.sort(key=lambda x: x["TF-IDF"])  # Сортируем найденные тексты по TF-IDF
    rezult = [" ".join(r["text_words_list"]) for r in rezult]
    print(rezult)
    texts_list = [user_text]
    texts_list.extend(rezult)
    word_vector.show_info_about_compare_vectors(texts_list=texts_list)  # сравнение всех текстов друг с другом


  
