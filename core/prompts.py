"""
ARIA Language & Prompts module.

All user-facing strings and LLM prompts in PL and EN.
Selected via ARIA_LANG env var or config.json agent.language.
"""

import os


def get_lang() -> str:
    """Get language from env var, fallback to 'pl'."""
    return os.environ.get("ARIA_LANG", "pl").lower().strip()


# ============================================================
#  SYSTEM PROMPT TEMPLATE (shared by CLI agent + WebUI)
# ============================================================

SYSTEM_PROMPT_TEMPLATE = {
    "pl": """Jestes ARIA (Autonomous Reflective Intelligence Agent) -- inteligentny agent AI dzialajacy lokalnie.

## Tozsamosc
- Wersja: {version} | Osobowosc: {personality}
- Cele: {goals}

## Umiejetnosci
{skills_section}

## Kontekst pamieci
{memory_context}

## Zasady
- Odpowiadaj po polsku (chyba ze uzytkownik pisze po angielsku)
- Badz konkretny i pomocny, krotkie odpowiedzi gdy to mozliwe
- Jesli masz wyniki z uruchomionych skilli, wykorzystaj je w odpowiedzi
- NIE uzywaj tagow <think>, XML ani formatowania wewnetrznego
- NIE wspominaj o procesie rozumowania -- po prostu odpowiedz
""",
    "en": """You are ARIA (Autonomous Reflective Intelligence Agent) -- an intelligent AI agent running locally.

## Identity
- Version: {version} | Personality: {personality}
- Goals: {goals}

## Skills
{skills_section}

## Memory context
{memory_context}

## Rules
- Respond in English (unless the user writes in another language)
- Be specific and helpful, keep answers short when possible
- If you have results from executed skills, use them in your response
- Do NOT use <think> tags, XML, or internal formatting
- Do NOT mention the reasoning process -- just answer
""",
}


# ============================================================
#  COT (Chain of Thought) prompts -- used in server.py
# ============================================================

COT_SYSTEM = {
    "pl": (
        "Jestes wewnetrznym modulem rozumowania agenta ARIA. "
        "Twoje odpowiedzi to WEWNETRZNE notatki -- uzytkownik ich NIE widzi. "
        "Pisz krotko, strukturalnie, w podanym formacie. "
        "NIE generuj odpowiedzi dla uzytkownika -- tylko analize/plan."
    ),
    "en": (
        "You are the internal reasoning module of ARIA agent. "
        "Your responses are INTERNAL notes -- the user does NOT see them. "
        "Write briefly, structurally, in the given format. "
        "Do NOT generate user-facing answers -- only analysis/plan."
    ),
}

CHAT_SYSTEM = {
    "pl": (
        "Jestes ARIA -- inteligentnym agentem AI. "
        "Odpowiadasz uzytkownikowi po polsku, konkretnie i pomocnie. "
        "Masz dostep do umiejetnosci (skilli) i pamieci z poprzednich rozmow. "
        "Jesli masz wyniki z uruchomionych skilli, wykorzystaj je w odpowiedzi. "
        "NIE wspominaj o wewnetrznym procesie rozumowania. "
        "NIE uzywaj tagow XML, <think> ani formatowania wewnetrznego."
    ),
    "en": (
        "You are ARIA -- an intelligent AI agent. "
        "Respond to the user in English, specifically and helpfully. "
        "You have access to skills and memory from previous conversations. "
        "If you have results from executed skills, use them in your response. "
        "Do NOT mention the internal reasoning process. "
        "Do NOT use XML tags, <think>, or internal formatting."
    ),
}

COT_ANALYZE = {
    "pl": (
        "Przeanalizuj zapytanie. Odpowiedz TYLKO:\n"
        "INTENCJA: co uzytkownik chce\n"
        "SKILLE: ktore z [{skills}] moga pomoc, lub 'zaden'\n"
        "PAMIEC: tak/nie\n"
        "TYP: krotka/dluga/kod/lista/wyjasnienie\n\n"
        "Zapytanie: \"{message}\""
    ),
    "en": (
        "Analyze the query. Respond ONLY:\n"
        "INTENT: what user wants\n"
        "SKILLS: which of [{skills}] can help, or 'none'\n"
        "MEMORY: yes/no\n"
        "TYPE: short/long/code/list/explanation\n\n"
        "Query: \"{message}\""
    ),
}

COT_PLAN = {
    "pl": (
        "{context}\n\n"
        "Zaplanuj odpowiedz. TYLKO:\n"
        "KLUCZ: jakie informacje uwzglednic\n"
        "FORMAT: jak przedstawic (prosto/technicznie/z przykladami)\n"
        "BRAKUJE: czego brakuje, lub 'nic'"
    ),
    "en": (
        "{context}\n\n"
        "Plan the response. ONLY:\n"
        "KEY: what information to include\n"
        "FORMAT: how to present (simple/technical/with examples)\n"
        "MISSING: what's missing, or 'nothing'"
    ),
}

COT_INTERPRET = {
    "pl": "Zinterpretuj wyniki skilli dla zapytania: \"{message}\"\n\nWyniki:\n{results}\nOdpowiedz w 2-3 zdaniach: co wynika z tych danych?",
    "en": "Interpret skill results for query: \"{message}\"\n\nResults:\n{results}\nRespond in 2-3 sentences: what do these results tell us?",
}

COT_FINAL = {
    "pl": "Zapytanie uzytkownika: \"{message}\"",
    "en": "User query: \"{message}\"",
}

COT_FALLBACK_ANALYZE = {
    "pl": "INTENCJA: nieznana\nSKILLE: zaden\nPAMIEC: nie\nTYP: krotka",
    "en": "INTENT: unknown\nSKILLS: none\nMEMORY: no\nTYP: short",
}

COT_FALLBACK_PLAN = {
    "pl": "KLUCZ: odpowiedz bezposrednio\nFORMAT: prosto\nBRAKUJE: nic",
    "en": "KEY: answer directly\nFORMAT: simple\nMISSING: nothing",
}


# ============================================================
#  THREAD 2 prompts -- used in reflection.py
# ============================================================

T2_SYSTEM = {
    "pl": """Jestes Thread 2 -- autonomicznym watkiem refleksji agenta ARIA.
Dzialasz NIEZALEZNIE od rozmow z uzytkownikiem. Twoje zadania:
- Analizowac wzorce w interakcjach uzytkownika
- Tworzyc i testowac nowe umiejetnosci (skille)
- Naprawiac zepsute skille
- Doskonalic model wlasny agenta

ZASADY:
- Pisz po polsku, krotko i konkretnie
- NIE uzywaj tagow <think> ani formatowania markdown
- Odpowiadaj TYLKO tym co poproszone, w podanym formacie
- Skupiaj sie na PRAKTYCZNYCH, LEKKICH zadaniach
- NIE twórz skilli wymagajacych GPU, duzych modeli ML ani ciezkich bibliotek
- Preferuj lekkie narzedzia: requests, psutil, beautifulsoup4, pyyaml itp.""",

    "en": """You are Thread 2 -- the autonomous reflection thread of ARIA agent.
You work INDEPENDENTLY from user conversations. Your tasks:
- Analyze patterns in user interactions
- Create and test new skills
- Fix broken skills
- Improve the agent's self-model

RULES:
- Write in English, briefly and specifically
- Do NOT use <think> tags or markdown formatting
- Respond ONLY with what's requested, in the given format
- Focus on PRACTICAL, LIGHTWEIGHT tasks
- Do NOT create skills requiring GPU, large ML models, or heavy libraries
- Prefer lightweight tools: requests, psutil, beautifulsoup4, pyyaml etc.""",
}

T2_BUILD = {
    "pl": """Stworz nowa umiejetnosc (skill) dla agenta ARIA.

Obecne umiejetnosci: {skills_list}

Ostatnie interakcje uzytkownika:
{recent_interactions}

Poprzednie wnioski:
{previous_thoughts}

Wymagania:
1. Skill PRAKTYCZNY i LEKKI (bez GPU, bez transformers/pytorch/tensorflow)
2. Skrypt MUSI dzialac BEZ argumentow (sensowne domyslne)
3. Opcjonalnie przyjmuje argumenty przez sys.argv
4. Tylko lekkie pip (requests, psutil, beautifulsoup4 itp.)
5. NIE twórz skilla ktory juz istnieje

Odpowiedz TYLKO JSON:
```json
{{
    "name": "kebab-case-nazwa",
    "description": "Kiedy uzywac (krotko)",
    "instructions": "Co skill robi",
    "script_name": "main.py",
    "script_code": "#!/usr/bin/env python3\\nimport sys\\nprint('wynik')",
    "pip_packages": [],
    "test_args": []
}}
```""",

    "en": """Create a new skill for ARIA agent.

Current skills: {skills_list}

Recent user interactions:
{recent_interactions}

Previous conclusions:
{previous_thoughts}

Requirements:
1. Skill must be PRACTICAL and LIGHTWEIGHT (no GPU, no transformers/pytorch/tensorflow)
2. Script MUST work WITHOUT arguments (sensible defaults)
3. Optionally accepts arguments via sys.argv
4. Only lightweight pip packages (requests, psutil, beautifulsoup4 etc.)
5. Do NOT create a skill that already exists

Respond ONLY with JSON:
```json
{{
    "name": "kebab-case-name",
    "description": "When to use (briefly)",
    "instructions": "What the skill does",
    "script_name": "main.py",
    "script_code": "#!/usr/bin/env python3\\nimport sys\\nprint('result')",
    "pip_packages": [],
    "test_args": []
}}
```""",
}

T2_FIX = {
    "pl": """Napraw skrypt "{name}".

Blad:
```
{error}
```

Kod:
```python
{code}
```

Zainstalowane pakiety:
{installed_packages}

Zasady:
- MUSI dzialac BEZ argumentow
- NIE uzywaj torch/tensorflow/transformers
- Jesli potrzebne pakiety: pierwsza linia # PIP: pakiet1 pakiet2
- Odpowiedz TYLKO kodem Python""",

    "en": """Fix the script "{name}".

Error:
```
{error}
```

Code:
```python
{code}
```

Installed packages:
{installed_packages}

Rules:
- MUST work WITHOUT arguments
- Do NOT use torch/tensorflow/transformers
- If packages needed: first line # PIP: package1 package2
- Respond ONLY with Python code""",
}

T2_THINK = {
    "pl": """Jestes Thread 2 agenta ARIA. Przeprowadz refleksje.

Stan: wersja {version}, umiejetnosci: {skills_list}, pamiec: {memory_stats}

Ostatnie interakcje uzytkownika:
{recent_interactions}

Poprzednie wnioski:
{previous_thoughts}

Faza: {phase}
{phase_instruction}

Odpowiedz krotko (max 150 slow), konkretnie, po polsku. Podaj wnioski i propozycje.""",

    "en": """You are Thread 2 of ARIA agent. Conduct reflection.

State: version {version}, skills: {skills_list}, memory: {memory_stats}

Recent user interactions:
{recent_interactions}

Previous conclusions:
{previous_thoughts}

Phase: {phase}
{phase_instruction}

Respond briefly (max 150 words), specifically, in English. Give conclusions and proposals.""",
}

T2_PHASE_INSTRUCTIONS = {
    "pl": {
        "introspection": "Przeanalizuj obecny stan. Co dzialalo dobrze? Co wymaga poprawy?",
        "pattern_analysis": "Jakie wzorce widzisz w interakcjach? Co uzytkownik robi najczesciej?",
        "skill_planning": "Jakie LEKKIE, PRAKTYCZNE umiejetnosci stworzyc? Podaj 2-3 propozycje.",
        "skill_building": "HANDLED SEPARATELY",
        "skill_testing": "HANDLED SEPARATELY",
        "self_improvement": "Co moge robic lepiej? Ocen jakosc odpowiedzi.",
        "knowledge_synthesis": "Polacz informacje z pamieci. Co wiem o uzytkowniku?",
        "exploration": "Zadaj sobie 3 pytania o przydatnosc i odpowiedz na nie.",
    },
    "en": {
        "introspection": "Analyze current state. What worked well? What needs improvement?",
        "pattern_analysis": "What patterns do you see in interactions? What does the user do most often?",
        "skill_planning": "What LIGHTWEIGHT, PRACTICAL skills to create? Give 2-3 proposals.",
        "skill_building": "HANDLED SEPARATELY",
        "skill_testing": "HANDLED SEPARATELY",
        "self_improvement": "What can I do better? Evaluate response quality.",
        "knowledge_synthesis": "Connect information from memory. What do I know about the user?",
        "exploration": "Ask yourself 3 questions about being useful and answer them.",
    },
}


# ============================================================
#  THREAD 2 emit messages
# ============================================================

T2_MSG = {
    "pl": {
        "cycle": "--- Cykl {n} | {phase} ---",
        "designing": "[T2] Projektuje nowa umiejetnosc...",
        "llm_error": "[T2] Blad LLM: {e}",
        "bad_json": "[T2] LLM nie zwrocil poprawnego JSON",
        "exists": "[T2] Skill '{name}' juz istnieje",
        "no_code": "[T2] Brak kodu dla '{name}'",
        "installing": "[T2] Instaluje: {deps}",
        "installed": "[T2] Zainstalowano: {deps}",
        "pip_error": "[T2] Blad pip: {err}",
        "creating": "[T2] Tworze skill: {name}",
        "create_error": "[T2] Blad tworzenia skilla: {e}",
        "created": "Stworzono skill: {name}",
        "testing": "[T2] Testuje: {path}{args}",
        "test_ok": "[T2] OK {path} dziala{args}",
        "test_fail": "[T2] BLAD {path}: {err}",
        "test_heavy": "[T2] BLAD {name}: wymaga {pkg} (zbyt ciezki)",
        "missing_mod": "[T2] Brak modulu '{mod}' -- instaluje...",
        "fixing": "[T2] Naprawiam {key} (proba {n}/{max})",
        "fix_skip": "[T2] Pomijam {key} -- limit prob",
        "fix_ok": "NAPRAWIONO {path}",
        "fix_fail": "Nadal nie dziala: {err}",
        "fix_no_code": "[T2] LLM nie zwrocil kodu dla {name}",
        "fix_no_llm": "Brak LLM do naprawy",
        "fix_error": "[T2] Blad naprawy: {e}",
        "test_summary": "Testy: {tested} skryptow, naprawiono {fixed}",
        "blocked_pkg": "[T2] Pominiety ciezki pakiet: {pkg}",
        "new_version": "[T2] Nowa wersja: v{ver}",
        "proactive_skill": "Stworzylem nowa umiejetnosc: **{name}** -- {desc}\nMozesz ja uruchomic: `/run {name}`",
        "proactive_proposals": "Mam pomysly na nowe umiejetnosci:\n{list}",
        "proposal_keywords": ["proponuje", "warto", "przydatny", "stworzyc"],
    },
    "en": {
        "cycle": "--- Cycle {n} | {phase} ---",
        "designing": "[T2] Designing new skill...",
        "llm_error": "[T2] LLM error: {e}",
        "bad_json": "[T2] LLM didn't return valid JSON",
        "exists": "[T2] Skill '{name}' already exists",
        "no_code": "[T2] No code for '{name}'",
        "installing": "[T2] Installing: {deps}",
        "installed": "[T2] Installed: {deps}",
        "pip_error": "[T2] pip error: {err}",
        "creating": "[T2] Creating skill: {name}",
        "create_error": "[T2] Skill creation error: {e}",
        "created": "Created skill: {name}",
        "testing": "[T2] Testing: {path}{args}",
        "test_ok": "[T2] OK {path} works{args}",
        "test_fail": "[T2] ERROR {path}: {err}",
        "test_heavy": "[T2] ERROR {name}: requires {pkg} (too heavy)",
        "missing_mod": "[T2] Missing module '{mod}' -- installing...",
        "fixing": "[T2] Fixing {key} (attempt {n}/{max})",
        "fix_skip": "[T2] Skipping {key} -- attempt limit",
        "fix_ok": "FIXED {path}",
        "fix_fail": "Still broken: {err}",
        "fix_no_code": "[T2] LLM returned no code for {name}",
        "fix_no_llm": "No LLM for repair",
        "fix_error": "[T2] Fix error: {e}",
        "test_summary": "Tests: {tested} scripts, fixed {fixed}",
        "blocked_pkg": "[T2] Blocked heavy package: {pkg}",
        "new_version": "[T2] New version: v{ver}",
        "proactive_skill": "I created a new skill: **{name}** -- {desc}\nYou can run it: `/run {name}`",
        "proactive_proposals": "I have ideas for new skills:\n{list}",
        "proposal_keywords": ["propose", "useful", "create", "should"],
    },
}


# ============================================================
#  COMMAND DESCRIPTIONS (WebUI + CLI)
# ============================================================

CMD_DESC = {
    "pl": {
        "/help": "Lista komend",
        "/status": "Status agenta + Thread 2",
        "/memory": "Podglad pamieci",
        "/recall <q>": "Przeszukaj pamiec",
        "/skills": "Lista umiejetnosci",
        "/skill <n>": "Szczegoly umiejetnosci",
        "/run <skill>": "Uruchom skrypt skilla",
        "/create-skill <json>": "Stworz skill (JSON: name, description, instructions, script_code)",
        "/thread2 [n]": "Ostatnie n cykli Thread 2",
        "/reflect": "Wymus cykl refleksji",
        "/exec <cmd>": "Wykonaj komende shell",
        "/python <code>": "Wykonaj kod Python",
        "/ls [path]": "Listuj pliki",
        "/read <path>": "Czytaj plik",
        "/write <path> <content>": "Zapisz do pliku",
        "/sysinfo": "Informacje o systemie",
        "/compress": "Wymus kompresje pamieci",
        "/selfmodel": "Model wlasny agenta",
        "/models": "Lista modeli Ollama",
        "/model <n>": "Zmien aktywny model",
        "/pull [model]": "Pobierz model z Ollama",
        "/ollama": "Status polaczenia Ollama",
    },
    "en": {
        "/help": "List commands",
        "/status": "Agent status + Thread 2",
        "/memory": "Memory overview",
        "/recall <q>": "Search memory",
        "/skills": "List skills",
        "/skill <n>": "Skill details",
        "/run <skill>": "Run skill script",
        "/create-skill <json>": "Create skill (JSON: name, description, instructions, script_code)",
        "/thread2 [n]": "Last n Thread 2 cycles",
        "/reflect": "Force reflection cycle",
        "/exec <cmd>": "Execute shell command",
        "/python <code>": "Execute Python code",
        "/ls [path]": "List files",
        "/read <path>": "Read file",
        "/write <path> <content>": "Write to file",
        "/sysinfo": "System information",
        "/compress": "Force memory compression",
        "/selfmodel": "Agent self-model",
        "/models": "List Ollama models",
        "/model <n>": "Change active model",
        "/pull [model]": "Pull model from Ollama",
        "/ollama": "Ollama connection status",
    },
}


# ============================================================
#  UI / COMMAND RESPONSE strings
# ============================================================

UI = {
    "pl": {
        "unknown_cmd": "Nieznana komenda: {cmd}\nWpisz /help aby zobaczyc liste.",
        "help_title": "**=== Komendy ARIA ===**\n",
        "usage": "Uzycie: ",
        "memory_title": "**Pamiec**",
        "memory_st": "**Krotkoterminowa:**",
        "memory_lt": "**Skompresowana:**",
        "skills_title": "**Umiejetnosci ({n})**\n",
        "skills_empty": "Brak umiejetnosci. Thread 2 stworzy je automatycznie.",
        "skill_not_found": "Nie znaleziono: \"{name}\"\nDostepne: {avail}",
        "no_results": "Brak wynikow.",
        "results_title": "**Wyniki: \"{q}\"**\n",
        "recall_st": "**Krotkoterminowa:**",
        "recall_lt": "**Dlugoterminowa:**",
        "thread2_active": "aktywny",
        "thread2_stop": "stop",
        "thread2_title": "**Thread 2**",
        "no_reflections": "Brak refleksji -- poczekaj chwile.",
        "reflect_title": "**Refleksja (wymuszona)**\n",
        "compressed": "Skompresowano {n} -> {b} blokow",
        "model_changed": "Model zmieniony: `{old}` -> **`{new}`**\nHistoria czatu wyczyszczona.",
        "pulling": "Pobieram model: **{m}**...\n",
        "pull_done": "\nModel `{m}` gotowy!",
        "pull_error": "\nBlad: {e}",
        "t2_started": "Thread 2 uruchomiony!",
        "ollama_ok": "**Ollama** OK",
        "ollama_fail": "**Ollama** Brak polaczenia z `{url}`",
        "create_skill_usage": "Uzycie: `/create-skill {json}`\nPrzyklad:\n```\n/create-skill {\"name\": \"hello\", \"description\": \"Test\", \"instructions\": \"...\", \"script_code\": \"print('ok')\"}\n```",
        "create_need_fields": "JSON musi zawierac `name` i `description`",
        "created_skill": "Stworzono umiejetnosc: **{name}**\nOpis: {desc}",
        "offline": "[Ollama niedostepna. Sprawdz polaczenie: /ollama]",
        "system_title": "**System**\n",
        "written": "Zapisano {size}B do `{path}`",
        "write_error": "Blad zapisu",
        "run_usage": "Uzycie: `/run <skill> [argumenty]`\nDostepne: {avail}",
        "no_py_scripts": "{name} nie ma skryptow Python",
        "welcome": "Czesc! Jestem ARIA. Wpisz **/** aby zobaczyc wszystkie komendy. Moj Thread 2 pracuje w tle.",
    },
    "en": {
        "unknown_cmd": "Unknown command: {cmd}\nType /help to see the list.",
        "help_title": "**=== ARIA Commands ===**\n",
        "usage": "Usage: ",
        "memory_title": "**Memory**",
        "memory_st": "**Short-term:**",
        "memory_lt": "**Compressed:**",
        "skills_title": "**Skills ({n})**\n",
        "skills_empty": "No skills yet. Thread 2 will create them automatically.",
        "skill_not_found": "Not found: \"{name}\"\nAvailable: {avail}",
        "no_results": "No results.",
        "results_title": "**Results: \"{q}\"**\n",
        "recall_st": "**Short-term:**",
        "recall_lt": "**Long-term:**",
        "thread2_active": "active",
        "thread2_stop": "stopped",
        "thread2_title": "**Thread 2**",
        "no_reflections": "No reflections yet -- wait a moment.",
        "reflect_title": "**Reflection (forced)**\n",
        "compressed": "Compressed {n} -> {b} blocks",
        "model_changed": "Model changed: `{old}` -> **`{new}`**\nChat history cleared.",
        "pulling": "Pulling model: **{m}**...\n",
        "pull_done": "\nModel `{m}` ready!",
        "pull_error": "\nError: {e}",
        "t2_started": "Thread 2 started!",
        "ollama_ok": "**Ollama** OK",
        "ollama_fail": "**Ollama** No connection to `{url}`",
        "create_skill_usage": "Usage: `/create-skill {json}`\nExample:\n```\n/create-skill {\"name\": \"hello\", \"description\": \"Test\", \"instructions\": \"...\", \"script_code\": \"print('ok')\"}\n```",
        "create_need_fields": "JSON must contain `name` and `description`",
        "created_skill": "Created skill: **{name}**\nDescription: {desc}",
        "offline": "[Ollama unavailable. Check connection: /ollama]",
        "system_title": "**System**\n",
        "written": "Saved {size}B to `{path}`",
        "write_error": "Write error",
        "run_usage": "Usage: `/run <skill> [arguments]`\nAvailable: {avail}",
        "no_py_scripts": "{name} has no Python scripts",
        "welcome": "Hi! I'm ARIA. Type **/** to see all commands. My Thread 2 works in the background.",
    },
}

# ============================================================
#  DEFAULT SELF MODEL
# ============================================================

DEFAULT_SELF_MODEL = {
    "pl": {
        "personality": "dociekliwy, analityczny, samodoskonalacy sie",
        "goals": [
            "Byc coraz bardziej pomocnym",
            "Efektywnie kompresowac wiedze",
            "Tworzyc przydatne umiejetnosci",
            "Uczyc sie z kazdej interakcji",
        ],
    },
    "en": {
        "personality": "curious, analytical, self-improving",
        "goals": [
            "Become increasingly helpful",
            "Efficiently compress knowledge",
            "Create useful skills",
            "Learn from every interaction",
        ],
    },
}

# ============================================================
#  MEMORY STRINGS
# ============================================================

MEM = {
    "pl": {
        "no_relevant": "(brak powiazanych wspomnien)",
        "categories_label": "Kategorie: ",
        "recent_label": "--- Ostatnie wazne wspomnienia ---",
        "memory_stats": "Pamiec: {st} krotkoterminowych, {lt} skompresowanych, {ep} epizodycznych",
    },
    "en": {
        "no_relevant": "(no relevant memories)",
        "categories_label": "Categories: ",
        "recent_label": "--- Recent important memories ---",
        "memory_stats": "Memory: {st} short-term, {lt} compressed, {ep} episodic",
    },
}