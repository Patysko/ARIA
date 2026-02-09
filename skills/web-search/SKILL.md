---
name: web-search
description: Wyszukiwanie w internecie przez SearXNG. Uzywaj gdy uzytkownik pyta o aktualne informacje, wiadomosci, dane z internetu, szukanie, google, wyszukaj, znajdz.
protected: true
---

# Web Search Skill

Umiejetnosc wyszukiwania w internecie za pomoca lokalnej instancji SearXNG.

## Uzycie

```bash
python3 scripts/web-search.py <zapytanie>
```

## Przyklady
- `python3 scripts/main.py "pogoda Warszawa"`
- `python3 scripts/main.py "najnowsze wiadomosci technologiczne"`
- `python3 scripts/main.py "python asyncio tutorial"`

## Konfiguracja
SearXNG dostepny pod adresem: http://host.docker.internal:8888
Format odpowiedzi: JSON z tytulami, URL-ami i opisami wynikow.