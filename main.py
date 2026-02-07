import aiohttp
import asyncio
import argparse
import sys
import pymorphy3
import text_tools
from adapters.inosmi_ru import sanitize
from adapters.exceptions import ArticleNotFound

# Настройки по умолчанию
FETCH_TIMEOUT = 5
ANALYZE_TIMEOUT = 3

def load_charged_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

async def fetch(session, url):
    timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT)
    async with session.get(url, timeout=timeout) as response:
        response.raise_for_status()
        return await response.text()

async def process_article(session, morph, charged_words, url):
    """Скачивает и анализирует одну статью."""
    try:
        html = await fetch(session, url)
        clean_text = sanitize(html, plaintext=True)
        
        # Выносим вычисления в поток, чтобы не блокировать Event Loop
        loop = asyncio.get_running_loop()
        score, words_count = await asyncio.wait_for(
            loop.run_in_executor(None, text_tools.split_by_words, morph, clean_text),
            timeout=ANALYZE_TIMEOUT
        )
        # Вторая часть вычислений (рейтинг)
        score = text_tools.calculate_jaundice_rate(score, charged_words)
        
        return {
            'url': url,
            'score': score,
            'words_count': len(words_count) if isinstance(words_count, list) else 0,
            'status': 'OK'
        }
    except asyncio.TimeoutError:
        return {'url': url, 'status': 'TIMEOUT'}
    except aiohttp.ClientError:
        return {'url': url, 'status': 'FETCH_ERROR'}
    except ArticleNotFound:
        return {'url': url, 'status': 'PARSING_ERROR'}
    except Exception as e:
        return {'url': url, 'status': f'ERROR: {e}'}

async def main():
    parser = argparse.ArgumentParser(description='Анализатор желтушности статей ИНОСМИ.РУ')
    parser.add_argument('urls', nargs='+', help='URL статьи(ий) для анализа')
    args = parser.parse_args()

    morph = pymorphy3.MorphAnalyzer()
    charged_words = load_charged_words('charged_dict/negative.txt')

    async with aiohttp.ClientSession() as session:
        tasks = [process_article(session, morph, charged_words, url) for url in args.urls]
        results = await asyncio.gather(*tasks)

        for res in results:
            print(f"\nURL: {res['url']}")
            if res['status'] == 'OK':
                print(f"Результат: {res['score']}% (слов: {res['words_count']})")
            else:
                print(f"Статус: {res['status']}")

if __name__ == '__main__':
    asyncio.run(main())
