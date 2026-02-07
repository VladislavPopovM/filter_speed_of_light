import asyncio
import logging
import time
from typing import Dict, Any

import aiohttp
import pymorphy3
from aiohttp import web

import text_tools
from adapters.exceptions import ArticleNotFound
from adapters.inosmi_ru import sanitize

# Настройки вынесены в константы для прозрачности
FETCH_TIMEOUT = 5
ANALYZE_TIMEOUT = 3
MAX_URLS = 10
# Семафор защищает внешние ресурсы от перегрузки (backpressure)
CONCURRENCY_LIMIT = asyncio.Semaphore(10)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def process_article(session: aiohttp.ClientSession, morph: pymorphy3.MorphAnalyzer, 
                          charged_words: list, url: str) -> Dict[str, Any]:
    """
    Весь цикл обработки статьи: скачивание -> очистка -> анализ.
    Возвращает плоский словарь с результатом и статусом.
    """
    try:
        # Ограничиваем количество одновременных сетевых запросов глобально
        async with CONCURRENCY_LIMIT:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=FETCH_TIMEOUT)) as resp:
                resp.raise_for_status()
                html = await resp.text()

        # Выделяем текст и считаем слова (тяжелые вычисления в пуле потоков)
        clean_text = sanitize(html, plaintext=True)
        
        loop = asyncio.get_running_loop()
        start_analysis = time.perf_counter()
        
        # Передаем только нужные данные в executor для минимизации накладных расходов
        words = await asyncio.wait_for(
            loop.run_in_executor(None, text_tools.split_by_words, morph, clean_text),
            timeout=ANALYZE_TIMEOUT
        )
        score = text_tools.calculate_jaundice_rate(words, charged_words)
        
        return {
            "url": url,
            "status": "OK",
            "score": score,
            "words_count": len(words),
            "duration": round(time.perf_counter() - start_analysis, 4)
        }

    except asyncio.TimeoutError:
        return {"url": url, "status": "TIMEOUT"}
    except aiohttp.ClientError:
        return {"url": url, "status": "FETCH_ERROR"}
    except ArticleNotFound:
        return {"url": url, "status": "PARSING_ERROR"}
    except Exception as e:
        logging.error(f"Error processing {url}: {e}")
        return {"url": url, "status": "INTERNAL_ERROR"}

async def analyze_handler(request: web.Request) -> web.Response:
    """Обработчик входящих запросов API."""
    urls_param = request.query.get('urls', '')
    urls = [u.strip() for u in urls_param.split(',') if u.strip()]

    if not urls:
        return web.json_response({"error": "No URLs provided"}, status=400)
    
    if len(urls) > MAX_URLS:
        return web.json_response({"error": f"Max {MAX_URLS} URLs allowed"}, status=400)

    # Запускаем задачи параллельно. Просто и эффективно.
    tasks = [
        process_article(request.app['session'], request.app['morph'], request.app['words'], url)
        for url in urls
    ]
    results = await asyncio.gather(*tasks)
    return web.json_response(results)

async def start_background_tasks(app: web.Application):
    """Инициализация ресурсов при старте сервера."""
    app['session'] = aiohttp.ClientSession()
    app['morph'] = pymorphy3.MorphAnalyzer()
    with open('charged_dict/negative.txt', 'r') as f:
        app['words'] = [line.strip() for line in f if line.strip()]
    yield
    await app['session'].close()

if __name__ == '__main__':
    app = web.Application()
    app.cleanup_ctx.append(start_background_tasks)
    app.add_routes([web.get('/api/analyze', analyze_handler)])
    web.run_app(app, port=8080)
