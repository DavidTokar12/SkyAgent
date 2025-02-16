from __future__ import annotations

from googlesearch import search


def google_search_tool(
    query: str,
    num_results: int = 10,
    lang: str = "en",
    region: str = "us",
    start_result: int = 0,
) -> dict:
    """
    Use this tool to perform a Google search.

    Args:
        query (str): The search query string (required)
        num_results (int): Number of results to return (default: 10)
        lang (str): Language code for search results (default: "en")
        region (str): Country code for search results (default: "us")

    Returns:
        dict: A dictionary containing the search results and an error message (if any).
    """
    try:

        search_results = search(
            query,
            advanced=True,
            num_results=num_results,
            lang=lang,
            region=region,
            unique=True,
            start_result=start_result,
        )

        formatted_results = {
            "results": [
                {
                    "title": result.title,
                    "description": result.description,
                    "url": result.url,
                }
                for result in search_results
            ],
            "error": None,
        }

        return formatted_results

    except Exception as e:
        return {"results": None, "error": str(e)}
