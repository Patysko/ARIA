#!/usr/bin/env python3
"""
Web Search via SearXNG — searches the internet and returns results.

Usage:
    python3 web-search.py <query>           # Search with query
    python3 web-search.py                   # Without args: prints help

SearXNG instance: http://host.docker.internal:8888
"""

import sys
import json
import urllib.request
import urllib.parse
import urllib.error

SEARXNG_URL = "http://host.docker.internal:8888"


def search(query: str, num_results: int = 5) -> list[dict]:
    """Search SearXNG and return results."""
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "engines": "google,duckduckgo,brave",
        "language": "pl-PL",
    })

    url = f"{SEARXNG_URL}/search?{params}"

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "ARIA-Agent/1.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = []
            for item in data.get("results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "engine": item.get("engine", ""),
                })
            return results
    except urllib.error.URLError as e:
        return [{"error": f"Blad polaczenia z SearXNG: {e}"}]
    except Exception as e:
        return [{"error": f"Blad wyszukiwania: {e}"}]


def main():
    if len(sys.argv) < 2:
        print("Web Search Skill - wyszukiwanie przez SearXNG")
        print(f"SearXNG URL: {SEARXNG_URL}")
        print("Uzycie: python3 web-search.py <zapytanie>")
        print("Przyklad: python3 web-search.py 'pogoda Warszawa'")

        # Check if SearXNG is available
        try:
            req = urllib.request.Request(f"{SEARXNG_URL}/")
            with urllib.request.urlopen(req, timeout=5):
                print("Status SearXNG: OK")
        except Exception:
            print("Status SearXNG: NIEDOSTEPNY")
        return

    query = " ".join(sys.argv[1:])
    results = search(query)

    if not results:
        print(f"Brak wynikow dla: {query}")
        return

    if results and "error" in results[0]:
        print(results[0]["error"])
        return

    print(f"Wyniki wyszukiwania: \"{query}\"\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}")
        print(f"   URL: {r['url']}")
        if r['content']:
            # Truncate content
            content = r['content'][:200]
            print(f"   {content}")
        print()


if __name__ == "__main__":
    main()