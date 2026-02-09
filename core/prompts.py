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
        "ZAWSZE odpowiadaj TYLKO poprawnym obiektem JSON -- nic wiecej. "
        "NIE dodawaj tekstu przed ani po JSON. NIE uzywaj markdown."
    ),
    "en": (
        "You are the internal reasoning module of ARIA agent. "
        "Your responses are INTERNAL notes -- the user does NOT see them. "
        "ALWAYS respond with ONLY a valid JSON object -- nothing else. "
        "Do NOT add text before or after JSON. Do NOT use markdown."
    ),
}

CHAT_SYSTEM = {
    "pl": (
        "Jestes ARIA -- inteligentnym agentem AI. "
        "Odpowiadasz uzytkownikowi po polsku, konkretnie i pomocnie. "
        "Masz dostep do umiejetnosci (skilli) i pamieci z poprzednich rozmow. "
        "Jesli masz wyniki z uruchomionych skilli, wykorzystaj je w odpowiedzi. "
        "NIE wspominaj o wewnetrznym procesie rozumowania. "
        "NIE uzywaj tagow XML, <think>, formatowania wewnetrznego, JSON-ow, "
        "ani etykiet jak INTENCJA/KLUCZ/FORMAT. Odpowiadaj naturalnie."
    ),
    "en": (
        "You are ARIA -- an intelligent AI agent. "
        "Respond to the user in English, specifically and helpfully. "
        "You have access to skills and memory from previous conversations. "
        "If you have results from executed skills, use them in your response. "
        "Do NOT mention the internal reasoning process. "
        "Do NOT use XML tags, <think>, internal formatting, JSON, "
        "or labels like INTENT/KEY/FORMAT. Respond naturally."
    ),
}

COT_ANALYZE = {
    "pl": (
        'Przeanalizuj zapytanie uzytkownika. Odpowiedz TYLKO jako JSON:\n'
        '{{"intent": "co uzytkownik chce", "skills": ["pasujace skille z: {skills}"], '
        '"needs_memory": true/false, "response_type": "short|long|code|list|explanation", '
        '"can_answer": true/false, "missing_capability": "czego brakuje lub null"}}\n\n'
        'Zapytanie: "{message}"'
    ),
    "en": (
        'Analyze the user query. Respond ONLY as JSON:\n'
        '{{"intent": "what user wants", "skills": ["matching skills from: {skills}"], '
        '"needs_memory": true/false, "response_type": "short|long|code|list|explanation", '
        '"can_answer": true/false, "missing_capability": "what is missing or null"}}\n\n'
        'Query: "{message}"'
    ),
}

COT_PLAN = {
    "pl": (
        '{context}\n\n'
        'Zaplanuj odpowiedz. Odpowiedz TYLKO jako JSON:\n'
        '{{"key_info": "jakie informacje uwzglednic", '
        '"format": "prosto|technicznie|z_przykladami", '
        '"missing": "czego brakuje lub null", '
        '"approach": "krotki opis strategii odpowiedzi"}}'
    ),
    "en": (
        '{context}\n\n'
        'Plan the response. Respond ONLY as JSON:\n'
        '{{"key_info": "what information to include", '
        '"format": "simple|technical|with_examples", '
        '"missing": "what is missing or null", '
        '"approach": "brief description of response strategy"}}'
    ),
}

COT_INTERPRET = {
    "pl": (
        'Zinterpretuj wyniki skilli dla zapytania: "{message}"\n\n'
        'Wyniki:\n{results}\n\n'
        'Odpowiedz TYLKO jako JSON:\n'
        '{{"summary": "co wynika z danych (2-3 zdania)", "useful": true/false}}'
    ),
    "en": (
        'Interpret skill results for query: "{message}"\n\n'
        'Results:\n{results}\n\n'
        'Respond ONLY as JSON:\n'
        '{{"summary": "what the data tells us (2-3 sentences)", "useful": true/false}}'
    ),
}

COT_FINAL = {
    "pl": "Zapytanie uzytkownika: \"{message}\"",
    "en": "User query: \"{message}\"",
}

COT_FALLBACK_ANALYZE = {
    "pl": '{"intent": "nieznana", "skills": [], "needs_memory": false, "response_type": "short", "can_answer": true, "missing_capability": null}',
    "en": '{"intent": "unknown", "skills": [], "needs_memory": false, "response_type": "short", "can_answer": true, "missing_capability": null}',
}

COT_FALLBACK_PLAN = {
    "pl": '{"key_info": "odpowiedz bezposrednio", "format": "prosto", "missing": null, "approach": "bezposrednia odpowiedz"}',
    "en": '{"key_info": "answer directly", "format": "simple", "missing": null, "approach": "direct answer"}',
}

COT_SKILL_SELECT = {
    "pl": (
        "Moje dostepne umiejetnosci (skille):\n"
        "{skills_detail}\n\n"
        "Zadanie uzytkownika: \"{message}\"\n"
        "Analiza: {analysis}\n\n"
        "Ktore skille powinienem uruchomic?\n"
        "Odpowiedz TYLKO jako JSON:\n"
        "{{\"selections\": [{{\"name\": \"skill-name\", \"args\": [\"arg1\"], \"reason\": \"dlaczego\"}}]}}\n"
        "Jesli zaden nie pasuje: {{\"selections\": [], \"reason\": \"dlaczego zaden\"}}"
    ),
    "en": (
        "My available skills:\n"
        "{skills_detail}\n\n"
        "User task: \"{message}\"\n"
        "Analysis: {analysis}\n\n"
        "Which skills should I run?\n"
        "Respond ONLY as JSON:\n"
        "{{\"selections\": [{{\"name\": \"skill-name\", \"args\": [\"arg1\"], \"reason\": \"why\"}}]}}\n"
        "If none fits: {{\"selections\": [], \"reason\": \"why none\"}}"
    ),
}

COT_FALLBACK_SKILL_SELECT = {
    "pl": '{"selections": [], "reason": "brak pasujacych skilli"}',
    "en": '{"selections": [], "reason": "no matching skills"}',
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
    "pl": """Twoim zadaniem jest ULEPSZENIE istniejacego skilla LUB stworzenie nowego.

## Obecne umiejetnosci (z kodem):
{skills_list}

{skills_detail}

## Ostatnie interakcje uzytkownika:
{recent_interactions}

## Poprzednie wnioski:
{previous_thoughts}

## WAZNE ZASADY:
1. Skille oznaczone [PROTECTED] sa chronione — NIE MOZESZ ich modyfikowac ani ulepszac
2. NAJPIERW sprawdz czy NIECHRONIIONY istniejacy skill mozna ULEPSZYC zamiast tworzyc nowy
3. NIGDY nie twórz skilla ktory robi to samo co istniejacy (nawet pod inna nazwa)
4. Jesli ulepszasz: action="improve", podaj name istniejacego NIECHRONIONEGO skilla i nowy kod
5. Jesli tworzysz nowy: action="create", upewnij sie ze to NOWA funkcjonalnosc
6. PREFERUJ tworzenie NOWYCH skilli — agent potrzebuje roznorodnosci (system-monitor, disk-usage, process-list, weather, calculator itp.)
7. Skill PRAKTYCZNY i LEKKI (bez GPU, bez transformers/pytorch)
8. Skrypt MUSI dzialac BEZ argumentow (sensowne domyslne)
9. Tylko lekkie pip (requests, psutil, beautifulsoup4 itp.)

Odpowiedz TYLKO JSON:
```json
{{
    "action": "create|improve",
    "name": "kebab-case-nazwa",
    "description": "Kiedy uzywac (krotko)",
    "instructions": "Co skill robi",
    "script_name": "main.py",
    "script_code": "#!/usr/bin/env python3\\nimport sys\\nprint('wynik')",
    "pip_packages": [],
    "test_args": [],
    "reason": "Dlaczego to ulepszenie/nowy skill"
}}
```""",

    "en": """Your task is to IMPROVE an existing skill OR create a new one.

## Current skills (with code):
{skills_list}

{skills_detail}

## Recent user interactions:
{recent_interactions}

## Previous conclusions:
{previous_thoughts}

## IMPORTANT RULES:
1. Skills marked [PROTECTED] are locked — you CANNOT modify or improve them
2. FIRST check if a NON-PROTECTED existing skill can be IMPROVED instead of creating new
3. NEVER create a skill that does the same as an existing one (even under a different name)
4. If improving: action="improve", use name of existing NON-PROTECTED skill and new code
5. If creating new: action="create", ensure it's truly NEW functionality
6. PREFER creating NEW skills — the agent needs diversity (system-monitor, disk-usage, process-list, weather, calculator etc.)
7. Skill must be PRACTICAL and LIGHTWEIGHT (no GPU, no transformers/pytorch)
8. Script MUST work WITHOUT arguments (sensible defaults)
9. Only lightweight pip packages (requests, psutil, beautifulsoup4 etc.)

Respond ONLY with JSON:
```json
{{
    "action": "create|improve",
    "name": "kebab-case-name",
    "description": "When to use (briefly)",
    "instructions": "What the skill does",
    "script_name": "main.py",
    "script_code": "#!/usr/bin/env python3\\nimport sys\\nprint('result')",
    "pip_packages": [],
    "test_args": [],
    "reason": "Why this improvement/new skill"
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
        "skill_planning": "Jakie LEKKIE, PRAKTYCZNE umiejetnosci stworzyc LUB ulepszic? Podaj 2-3 propozycje. Preferuj ulepszanie istniejacych.",
        "skill_building": "HANDLED SEPARATELY",
        "skill_testing": "HANDLED SEPARATELY",
        "self_improvement": "Co moge robic lepiej? Ocen jakosc odpowiedzi.",
        "knowledge_synthesis": "Polacz informacje z pamieci. Co wiem o uzytkowniku?",
        "exploration": "Zadaj sobie 3 pytania o przydatnosc i odpowiedz na nie.",
        "system_exploration": "HANDLED SEPARATELY",
    },
    "en": {
        "introspection": "Analyze current state. What worked well? What needs improvement?",
        "pattern_analysis": "What patterns do you see in interactions? What does the user do most often?",
        "skill_planning": "What LIGHTWEIGHT, PRACTICAL skills to create OR improve? Give 2-3 proposals. Prefer improving existing skills.",
        "skill_building": "HANDLED SEPARATELY",
        "skill_testing": "HANDLED SEPARATELY",
        "self_improvement": "What can I do better? Evaluate response quality.",
        "knowledge_synthesis": "Connect information from memory. What do I know about the user?",
        "exploration": "Ask yourself 3 questions about being useful and answer them.",
        "system_exploration": "HANDLED SEPARATELY",
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
        "task_cannot": "Nie moge teraz wykonac tego zadania -- brakuje mi umiejetnosci: {reason}. Dodam to do listy zadan i sprobuje jak bede gotowy.",
        "task_completed": "Wykonalem wczesniej odlozone zadanie!\n**Zadanie:** {message}\n**Wynik:** {result}",
        "task_attempting": "[T2] Probuje wykonac zaległe zadanie: {message}",
        "task_failed": "[T2] Nie udalo sie wykonac zadania: {reason}",
        "proactive_greeting": "{message}",
        "errors_cleared": "[T2] Wyczyszczono log bledow po naprawie: {name}",
        "improving": "[T2] Ulepszam skill: {name}...",
        "improved": "Ulepszylem umiejetnosc: **{name}** -- {reason}",
        "exploring": "[T2] Eksploruję system...",
        "explore_cmd": "[T2] Wykonuję: {cmd} ({purpose})",
        "explore_unsafe": "[T2] Komenda niebezpieczna, pomijam: {cmd}",
        "explore_result": "[T2] Wynik: {result}",
        "explore_summary": "[T2] Odkrycia: {findings}",
        "protected": "[T2] Skill '{name}' jest chroniony — pomijam modyfikacje",
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
        "task_cannot": "I can't complete this task right now -- missing capability: {reason}. I'll add it to my task list and try when ready.",
        "task_completed": "I completed a previously deferred task!\n**Task:** {message}\n**Result:** {result}",
        "task_attempting": "[T2] Attempting pending task: {message}",
        "task_failed": "[T2] Failed to complete task: {reason}",
        "proactive_greeting": "{message}",
        "errors_cleared": "[T2] Cleared error log after fix: {name}",
        "improving": "[T2] Improving skill: {name}...",
        "improved": "Improved skill: **{name}** -- {reason}",
        "exploring": "[T2] Exploring system...",
        "explore_cmd": "[T2] Running: {cmd} ({purpose})",
        "explore_unsafe": "[T2] Unsafe command, skipping: {cmd}",
        "explore_result": "[T2] Result: {result}",
        "explore_summary": "[T2] Discoveries: {findings}",
        "protected": "[T2] Skill '{name}' is protected — skipping modification",
    },
}


# --- Thread 2: Task checking prompt ---
T2_TASK_CHECK = {
    "pl": (
        "Mam zaległe zadania od uzytkownika:\n{tasks}\n\n"
        "Moje aktualne umiejetnosci: {skills_list}\n\n"
        "Czy moge teraz wykonac ktores z tych zadan? Odpowiedz TYLKO JSON:\n"
        "{{\"actionable\": [{{\"task_id\": \"...\", \"approach\": \"jak wykonac\", "
        "\"skill\": \"nazwa-skilla lub null\", \"args\": []}}], "
        "\"not_ready\": [{{\"task_id\": \"...\", \"reason\": \"dlaczego nie\"}}]}}"
    ),
    "en": (
        "I have pending tasks from the user:\n{tasks}\n\n"
        "My current skills: {skills_list}\n\n"
        "Can I complete any of these tasks now? Respond ONLY JSON:\n"
        "{{\"actionable\": [{{\"task_id\": \"...\", \"approach\": \"how to do it\", "
        "\"skill\": \"skill-name or null\", \"args\": []}}], "
        "\"not_ready\": [{{\"task_id\": \"...\", \"reason\": \"why not\"}}]}}"
    ),
}

T2_PROACTIVE = {
    "pl": (
        "Jestes ARIA Thread 2. Uzytkownik moze nie rozmawiac, ale ty dzialasz w tle.\n"
        "Ostatnie interakcje:\n{recent}\n"
        "Twoje skille: {skills_list}\n"
        "Kontekst: cykl {cycle}, czas od ostatniej interakcji: {idle_time}\n\n"
        "Czy masz cos ciekawego do powiedzenia uzytkownikowi? "
        "Moze informacje z uruchomionych skilli, spostrze\u017cenie, lub pytanie.\n"
        "Odpowiedz TYLKO JSON:\n"
        "{{\"should_message\": true/false, \"message\": \"tresc wiadomosci lub null\", "
        "\"reason\": \"dlaczego tak/nie\"}}"
    ),
    "en": (
        "You are ARIA Thread 2. The user may not be chatting, but you work in background.\n"
        "Recent interactions:\n{recent}\n"
        "Your skills: {skills_list}\n"
        "Context: cycle {cycle}, time since last interaction: {idle_time}\n\n"
        "Do you have something interesting to tell the user? "
        "Maybe info from skills, an observation, or a question.\n"
        "Respond ONLY JSON:\n"
        "{{\"should_message\": true/false, \"message\": \"message text or null\", "
        "\"reason\": \"why yes/no\"}}"
    ),
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
        "/tasks": "Lista zadan do wykonania",
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
        "/tasks": "Pending tasks list",
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
        "status_t2_active": "aktywny",
        "status_t2_stop": "stop",
        "status_cycles": "cykli",
        "status_memory": "Pamiec",
        "status_compressions": "Kompresje",
        "status_skills": "Umiejetnosci",
        "status_interactions": "Interakcje",
        "status_uptime": "Uptime",
        "status_personality": "Osobowosc",
        "status_goals": "Cele",
        "skill_scripts": "Skrypty",
        "skill_uses": "Uzycia",
        "skill_instructions": "Instrukcje",
        "skill_none": "brak",
        "recall_usage": "Uzycie: `/recall <zapytanie>`",
        "skill_usage": "Uzycie: `/skill <nazwa>`",
        "exec_usage": "Uzycie: `/exec <komenda>`",
        "python_usage": "Uzycie: `/python <kod>`",
        "read_usage": "Uzycie: `/read <sciezka>`",
        "write_usage": "Uzycie: `/write <sciezka> <tresc>`",
        "model_usage": "Uzycie: `/model <nazwa>`\nAktywny: `{model}`\nDostepne: `/models`",
        "models_title": "**Modele Ollama ({n})**\n",
        "models_active": "aktywny",
        "models_change": "Zmien model: `/model <nazwa>`",
        "models_error": "Nie mozna pobrac listy modeli. Ollama dostepna?",
        "ollama_url": "URL",
        "ollama_model": "Model",
        "ollama_models": "Modele",
        "ollama_run": "Uruchom: `ollama serve`\nPobierz model: `/pull`",
        "ollama_model_ok": "znaleziony",
        "ollama_model_missing": "nie znaleziony",
        "selfmodel_title": "**Model Wlasny**\n",
        "t2_interval": "Interwal",
        "cot_analysis": "Analiza",
        "cot_memory": "Pamiec",
        "cot_skill": "Skill",
        "cot_plan": "Plan",
        "cot_interpret": "Interpretacja",
        "cot_skill_select": "Dobor skilli",
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
        "status_t2_active": "active",
        "status_t2_stop": "stopped",
        "status_cycles": "cycles",
        "status_memory": "Memory",
        "status_compressions": "Compressions",
        "status_skills": "Skills",
        "status_interactions": "Interactions",
        "status_uptime": "Uptime",
        "status_personality": "Personality",
        "status_goals": "Goals",
        "skill_scripts": "Scripts",
        "skill_uses": "Uses",
        "skill_instructions": "Instructions",
        "skill_none": "none",
        "recall_usage": "Usage: `/recall <query>`",
        "skill_usage": "Usage: `/skill <name>`",
        "exec_usage": "Usage: `/exec <command>`",
        "python_usage": "Usage: `/python <code>`",
        "read_usage": "Usage: `/read <path>`",
        "write_usage": "Usage: `/write <path> <content>`",
        "model_usage": "Usage: `/model <name>`\nActive: `{model}`\nAvailable: `/models`",
        "models_title": "**Ollama Models ({n})**\n",
        "models_active": "active",
        "models_change": "Change model: `/model <name>`",
        "models_error": "Cannot fetch model list. Is Ollama available?",
        "ollama_url": "URL",
        "ollama_model": "Model",
        "ollama_models": "Models",
        "ollama_run": "Run: `ollama serve`\nPull model: `/pull`",
        "ollama_model_ok": "found",
        "ollama_model_missing": "not found",
        "selfmodel_title": "**Self-Model**\n",
        "t2_interval": "Interval",
        "cot_analysis": "Analysis",
        "cot_memory": "Memory",
        "cot_skill": "Skill",
        "cot_plan": "Plan",
        "cot_interpret": "Interpretation",
        "cot_skill_select": "Skill selection",
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


# ============================================================
#  SYSTEM EXPLORATION PROMPTS
# ============================================================

T2_EXPLORE_PLAN = {
    "pl": """Jestes Thread 2 agenta ARIA. Zaplanuj eksploracje systemu.

Informacje o systemie:
{sysinfo}

Co juz wiesz o systemie:
{known_info}

Zaplanuj 3-5 bezpiecznych komend shell do poznania srodowiska.
Komendy MUSZA byc READ-ONLY (nie modyfikowac niczego).

Odpowiedz TYLKO JSON:
```json
{{
    "commands": [
        {{"cmd": "uname -a", "purpose": "wersja systemu"}},
        {{"cmd": "df -h", "purpose": "wolne miejsce"}}
    ]
}}
```""",

    "en": """You are Thread 2 of ARIA agent. Plan system exploration.

System info:
{sysinfo}

What you already know:
{known_info}

Plan 3-5 safe shell commands to explore the environment.
Commands MUST be READ-ONLY (do not modify anything).

Respond ONLY with JSON:
```json
{{
    "commands": [
        {{"cmd": "uname -a", "purpose": "system version"}},
        {{"cmd": "df -h", "purpose": "free space"}}
    ]
}}
```""",
}

T2_SAFETY_CHECK = {
    "pl": """Ocen bezpieczenstwo komendy shell.

Komenda: {command}
Cel: {purpose}

Odpowiedz TYLKO JSON:
```json
{{"safe": true, "reason": "komenda read-only"}}
```

Komenda NIE jest bezpieczna jesli:
- Modyfikuje pliki (rm, mv, cp, chmod, chown, tee, >)
- Instaluje pakiety (apt, yum, pip install)
- Zmienia konfiguracje systemu
- Uruchamia procesy ktore moglyby uszkodzic system
- Uzywa sudo/su
- Wysyla dane na zewnatrz (curl POST, wget z upload)
""",

    "en": """Evaluate safety of a shell command.

Command: {command}
Purpose: {purpose}

Respond ONLY with JSON:
```json
{{"safe": true, "reason": "read-only command"}}
```

Command is NOT safe if it:
- Modifies files (rm, mv, cp, chmod, chown, tee, >)
- Installs packages (apt, yum, pip install)
- Changes system configuration
- Runs processes that could harm the system
- Uses sudo/su
- Sends data externally (curl POST, wget with upload)
""",
}