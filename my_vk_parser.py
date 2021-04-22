import random
import requests
import text_search
import word_vector


def get_text_from_vk(*, query, domain, count):
    token_auth = '61249f3161249f3161249f31aa615382936612461249f310145a604685c59e9ecda0d4b'   # тут нужен токен авторизаци
    version = '5.130'
    response = requests.get('https://api.vk.com/method/wall.search',
                            params={
                                "access_token": token_auth,
                                "v": version,
                                "count": count,
                                "domain": domain,
                                "query": query,  # Может лучше искать отдельно по словам?
                            }
                            )
    data = response.json()["response"]["items"]
    texts_from_vk = [d["text"] for d in data]
    return texts_from_vk

def user_chouse_emulator(text_list):  # Имитация выбора пользователя
    return random.choice(text_list)




if __name__ == "__main__":
    key_info = "движение вперёд"
    texts_list = get_text_from_vk(query=key_info, count=5, domain='sports_books')    # поиск постов с ключевыми словами
    print(texts_list)

    user_text = user_chouse_emulator(texts_list)  # выбор текста пользователем (один случайный текст)
    for bad_symbol in [',', '.', '!', '?']:  # выкинуть из текста символы пунтуации, чтобы не было слов в стиле "Я,"
        key_info.replace(bad_symbol, '')
        user_text.replace(bad_symbol, '')

    key_list = key_info.split()  # Разбиение по пробелам
    words_list = user_text.split()
    new_key_list = []
    for key in key_list:  # поиск новых ключевых слов
        new_key_list.extend(text_search.frequancy_analis(words_list=words_list, key_word=key, max_new_words=2))
    print(new_key_list)
    print(len(new_key_list))
    new_texts_list = get_text_from_vk(query=key_info, count=2, domain='sports_books')    # поиск постов с расширенным
                                                                                         # набором ключевых слов
    print(new_texts_list)

    rezult = text_search.search(key_info=key_info, texts_list=new_texts_list)  # расчет TF-IDF
    rezult.sort(key=lambda x: x["TF-IDF"])
    print(rezult)

    word_vector.show_info_about_compare_vectors([user_text, " ".join(rezult[0]["text_words_list"])])
