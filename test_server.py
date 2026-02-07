import pytest
import aiohttp
import asyncio
import pymorphy3
from server import process_article
from adapters.exceptions import ArticleNotFound

# Фикстура для морфологического анализатора (создаем один раз)
@pytest.fixture(scope="session")
def morph():
    return pymorphy3.MorphAnalyzer()

# Фикстура для словаря
@pytest.fixture
def charged_words():
    return ['плохой', 'ужасный', 'шок']

@pytest.mark.asyncio
async def test_process_article_success(morph, charged_words):
    # Тест успешной обработки (нужна живая ссылка или мок)
    # Для простоты используем реальную ссылку, которая точно работает
    async with aiohttp.ClientSession() as session:
        url = 'https://inosmi.ru/20190629/245384784.html'
        result = await process_article(session, morph, charged_words, url)
        assert result['status'] == 'OK'
        assert result['words_count'] > 0
        assert 'score' in result

@pytest.mark.asyncio
async def test_process_article_not_found(morph, charged_words):
    async with aiohttp.ClientSession() as session:
        url = 'https://inosmi.ru/not-existent-article'
        result = await process_article(session, morph, charged_words, url)
        # В нашей логике 404 превращается либо в FETCH_ERROR (из-за raise_for_status)
        # либо в PARSING_ERROR, если sanitize не нашел статью.
        assert result['status'] in ['FETCH_ERROR', 'PARSING_ERROR']

@pytest.mark.asyncio
async def test_process_article_timeout(morph, charged_words):
    async with aiohttp.ClientSession() as session:
        # Ссылка, которая точно будет грузиться долго
        url = 'https://httpbin.org/delay/10'
        # У нас в server.py таймаут 5 секунд, так что 10 секунд точно вызовут ошибку
        result = await process_article(session, morph, charged_words, url)
        assert result['status'] == 'TIMEOUT'
