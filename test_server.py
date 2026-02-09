import pytest
import aiohttp
from concurrent.futures import ProcessPoolExecutor
from server import process_article


@pytest.fixture(scope="session")
def executor():
    executor = ProcessPoolExecutor(max_workers=2)
    yield executor
    executor.shutdown()


@pytest.fixture
def charged_words():
    return ["плохой", "ужасный", "шок", "трамп"]


@pytest.mark.asyncio
async def test_process_article_success(executor, charged_words):
    async with aiohttp.ClientSession() as session:
        url = "https://inosmi.ru/20190629/245384784.html"
        result = await process_article(session, executor, charged_words, url)
        assert result["status"] == "OK"
        assert result["words_count"] > 0
        assert isinstance(result["score"], float)


@pytest.mark.asyncio
async def test_process_article_not_found(executor, charged_words):
    async with aiohttp.ClientSession() as session:
        url = "https://inosmi.ru/non-existent"
        result = await process_article(session, executor, charged_words, url)
        assert result["status"] in ["FETCH_ERROR", "PARSING_ERROR"]


@pytest.mark.asyncio
async def test_process_article_timeout(executor, charged_words):
    async with aiohttp.ClientSession() as session:
        url = "https://httpbin.org/delay/10"
        result = await process_article(session, executor, charged_words, url)
        assert result["status"] == "TIMEOUT"
