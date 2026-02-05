"""Web Search Skill - Search the web via DuckDuckGo HTML (no API key)."""

import logging
import re
from urllib.parse import quote_plus

from openclaw.skills.base import BaseSkill

logger = logging.getLogger("openclaw.skills.web_search")


class WebSearchSkill(BaseSkill):
    name = "web_search"
    description = "Search the web for information using DuckDuckGo"
    tags = ["search", "web", "internet", "query", "find", "lookup"]

    async def execute(self, **kwargs) -> dict:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)

        if not query:
            return {"success": False, "error": "No query provided"}

        try:
            import httpx
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

            async with httpx.AsyncClient(
                timeout=15,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; OpenClaw/1.0)"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

            results = self._parse_results(response.text, max_results)

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
            }

        except ImportError:
            return {"success": False, "error": "httpx not installed. Run: pip install httpx"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_results(self, html: str, max_results: int) -> list[dict]:
        """Parse DuckDuckGo HTML results."""
        results = []

        # Find result blocks
        result_blocks = re.findall(
            r'<a rel="nofollow" class="result__a" href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a class="result__snippet"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        )

        for url, title, snippet in result_blocks[:max_results]:
            title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
            })

        return results
