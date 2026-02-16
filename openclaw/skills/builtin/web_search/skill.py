"""Web Search Skill - Search the web via SearXNG (preferred) or DuckDuckGo fallback."""

import logging
import re
from urllib.parse import quote_plus

from openclaw.skills.base import BaseSkill
from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.skills.web_search")


class WebSearchSkill(BaseSkill):
    name = "web_search"
    description = "Search the web for information using SearXNG or DuckDuckGo"
    tags = ["search", "web", "internet", "query", "find", "lookup"]

    def __init__(self, skill_path=None):
        super().__init__(skill_path=skill_path)
        self.settings = get_settings()

    async def execute(self, **kwargs) -> dict:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        engines = kwargs.get("engines", None)  # Optional: google,bing,duckduckgo

        if not query:
            return {"success": False, "error": "No query provided"}

        # Try SearXNG first if configured
        searxng_url = self.settings.get("skills.web_search.searxng_url", "http://searxng:8080")
        if searxng_url:
            result = await self._search_searxng(query, max_results, engines, searxng_url)
            if result["success"]:
                return result
            logger.warning(f"SearXNG failed, falling back to DuckDuckGo: {result.get('error')}")

        # Fallback to DuckDuckGo HTML scraping
        return await self._search_duckduckgo(query, max_results)

    async def _search_searxng(self, query: str, max_results: int, engines: str, base_url: str) -> dict:
        """Search using SearXNG JSON API."""
        try:
            import httpx

            params = {
                "q": query,
                "format": "json",
                "safesearch": 1,
            }
            if engines:
                params["engines"] = engines

            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(f"{base_url}/search", params=params)
                response.raise_for_status()
                data = response.json()

            results = []
            for item in data.get("results", [])[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "engine": item.get("engine", ""),
                })

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
                "source": "searxng",
                "infoboxes": data.get("infoboxes", []),
                "suggestions": data.get("suggestions", []),
            }

        except ImportError:
            return {"success": False, "error": "httpx not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _search_duckduckgo(self, query: str, max_results: int) -> dict:
        """Fallback: Search using DuckDuckGo HTML scraping."""
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

            results = self._parse_ddg_results(response.text, max_results)

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
                "source": "duckduckgo",
            }

        except ImportError:
            return {"success": False, "error": "httpx not installed. Run: pip install httpx"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_ddg_results(self, html: str, max_results: int) -> list[dict]:
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
                "engine": "duckduckgo",
            })

        return results
