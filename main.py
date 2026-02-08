#!/usr/bin/env python3
"""
ARIA â€” Autonomous Reflective Intelligence Agent

Usage:
    python web-search.py                  # CLI interactive mode
    python web-search.py --web            # WebUI on port 8080
    python web-search.py --web --port 3000
    python web-search.py --reflection     # CLI with Thread 2 live output
    python web-search.py --status         # Show agent status
"""

import sys
import os
import signal
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="ARIA â€” Autonomous Reflective Intelligence Agent")
    parser.add_argument("--web", action="store_true", help="Start WebUI instead of CLI")
    parser.add_argument("--port", type=int, default=8080, help="WebUI port (default: 8080)")
    parser.add_argument("--reflection", action="store_true", help="Show reflection thread live (CLI)")
    parser.add_argument("--status", action="store_true", help="Show agent status and exit")
    parser.add_argument("--skills", action="store_true", help="List all skills and exit")
    parser.add_argument("--memory", action="store_true", help="Dump memory and exit")
    parser.add_argument("--reset", action="store_true", help="Reset agent memory")
    args = parser.parse_args()

    if args.web:
        os.environ["ARIA_PORT"] = str(args.port)
        from web.server import main as web_main
        web_main()
        return

    from core.agent import AriaAgent
    from core.config import Config

    config = Config()
    agent = AriaAgent(config)

    if args.reset:
        agent.memory.reset()
        print("ðŸ—‘ï¸  PamiÄ™Ä‡ agenta zostaÅ‚a zresetowana.")
        return
    if args.status:
        agent.print_status()
        return
    if args.skills:
        agent.print_skills()
        return
    if args.memory:
        agent.print_memory_dump()
        return

    def signal_handler(sig, frame):
        print("\n\nðŸ”´ ZapisujÄ™ pamiÄ™Ä‡...")
        agent.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    agent.start(show_reflection=args.reflection)


if __name__ == "__main__":
    main()