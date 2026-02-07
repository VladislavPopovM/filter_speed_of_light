import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Tuple

import aiohttp
import pymorphy3
from aiohttp import web

import text_tools
from adapters.exceptions import ArticleNotFound
from adapters.inosmi_ru import sanitize

# Настройки
FETCH_TIMEOUT = 5
ANALYZE_TIMEOUT = 3
MAX_URLS = 10
CONCURRENCY_LIMIT = asyncio.Semaphore(10)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Глобальный кэш для морфолога внутри воркера (чтобы не инициализировать на каждый чих)
_morph_analyzer = None

def _get_morph():
    global _morph_analyzer
    if _morph_analyzer is None:
        _morph_analyzer = pymorphy3.MorphAnalyzer()
    return _morph_analyzer

def analyze_text_task(charged_words: List[str], text: str) -> Tuple[float, int]:
    """
    Тяжелая CPU-bound задача. Выполняется в отдельном ПРОЦЕССЕ.
    """
    morph = _get_morph()
    words = text_tools.split_by_words(morph, text)
    score = text_tools.calculate_jaundice_rate(words, charged_words)
    return score, len(words)

async def process_article(session: aiohttp.ClientSession, executor: ProcessPoolExecutor, 
                          charged_words: list, url: str) -> Dict[str, Any]:
    try:
        async with CONCURRENCY_LIMIT:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT)) as resp:
                resp.raise_for_status()
                html = await resp.text()

        clean_text = sanitize(html, plaintext=True)
        
        loop = asyncio.get_running_loop()
        start_analysis = time.perf_counter()
        
        # Настоящий параллелизм: уходим из основного процесса в пул воркеров
        score, words_count = await asyncio.wait_for(
            loop.run_in_executor(executor, analyze_text_task, charged_words, clean_text),
            timeout=ANALYZE_TIMEOUT
        )
        
        return {
            "url": url,
            "status": "OK",
            "score": score,
            "words_count": words_count,
            "duration": round(time.perf_counter() - start_analysis, 4)
        }

    except asyncio.TimeoutError:
        return {"url": url, "status": "TIMEOUT"}
    except (aiohttp.ClientError, asyncio.exceptions.TimeoutError):
        return {"url": url, "status": "FETCH_ERROR"}
    except ArticleNotFound:
        return {"url": url, "status": "PARSING_ERROR"}
    except Exception as e:
        logging.error(f"Error processing {url}: {e}")
        return {"url": url, "status": "INTERNAL_ERROR"}

async def analyze_handler(request: web.Request) -> web.Response:
    urls_param = request.query.get('urls', '')
    urls = [u.strip() for u in urls_param.split(',') if u.strip()]

    if not urls:
        return web.json_response({"error": "No URLs provided"}, status=400)
    
    if len(urls) > MAX_URLS:
        return web.json_response({"error": f"Max {MAX_URLS} URLs allowed"}, status=400)

    # Запускаем задачи параллельно в пуле процессов
    tasks = [
        process_article(request.app['session'], request.app['executor'], request.app['words'], url)
        for url in urls
    ]
    results = await asyncio.gather(*tasks)
    return web.json_response(results)

async def start_background_tasks(app: web.Application):
    """Инициализация ресурсов при старте сервера."""
    app['session'] = aiohttp.ClientSession()
    # Создаем пул процессов по количеству ядер CPU
    app['executor'] = ProcessPoolExecutor()
    with open('charged_dict/negative.txt', 'r') as f:
        app['words'] = [line.strip() for line in f if line.strip()]
    yield
    # Корректно завершаем сессию и пул процессов
    await app['session'].close()
    app['executor'].shutdown()

if __name__ == '__main__':
    app = web.Application()
    app.cleanup_ctx.append(start_background_tasks)
    app.add_routes([web.get('/api/analyze', analyze_handler)])
    web.run_app(app, port=8080)