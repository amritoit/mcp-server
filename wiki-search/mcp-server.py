import wikipedia
from mcp.server.fastmcp import FastMCP
from pathlib import Path


mcp = FastMCP("WikipediaSearch")

import argparse
from logger import setup_logger  # We'll define this next

parser = argparse.ArgumentParser()
parser.add_argument("--log", type=str, default="logs/server.log")
args = parser.parse_args()

logger = setup_logger("mcp-server", args.log)
#logger.debug("Server started")


@mcp.resource("file://suggested_titles")
def suggested_titles() -> list[str]:
    """
    Read and return suggested Wikipedia topics from a local file.
    """
    try:
        path = Path("suggested_titles.txt")
        if not path.exists():
            return ["File not found"]
        return path.read_text(encoding="utf-8").strip().splitlines()
    except Exception as e:
        return [f"Error reading file: {str(e)}"]
    

@mcp.prompt()
def highlight_sections_prompt(topic: str) -> str:
    """
    Identifies the most important sections from a Wikipedia article on the given topic.
    """
    return f"""
    The user is exploring the Wikipedia article on "{topic}".

    Given the list of section titles from the article, choose the 3–5 most important or interesting sections 
    that are likely to help someone learn about the topic.

    Return a bullet list of these section titles, along with 1-line explanations of why each one matters.
    """

@mcp.tool()
def fetch_wikipedia_info(query: str) -> dict:
    """
    Search Wikipedia for a topic and return title, summary, and URL of the best match.
    """
    try:
        logger.info('Fetching wikipedia for search:{query}')
        
        search_results = wikipedia.search(query)
        if not search_results:
            return {"error": "No results found for your query."}

        best_match = search_results[0]
        page = wikipedia.page(best_match)

        return {
            "title": page.title,
            "summary": page.summary,
            "url": page.url
        }

    except wikipedia.DisambiguationError as e:
        return {
            "error": f"Ambiguous topic. Try one of these: {', '.join(e.options[:5])}"
        }

    except wikipedia.PageError:
        return {
            "error": "No Wikipedia page could be loaded for this query."
        }


@mcp.tool()
def list_wikipedia_sections(topic: str) -> dict:
    """
    Return a list of section titles from the Wikipedia page of a given topic.
    """
    try:
        logger.info('Listing wikipedia sections for topic:{topic}')       
        page = wikipedia.page(topic)
        sections = page.sections
        return {"sections": sections}
    except Exception as e:
        return {"error": str(e)}
    

@mcp.tool()
def get_section_content(topic: str, section_title: str) -> dict:
    """
    Return the content of a specific section in a Wikipedia article.
    """
    try:
        logger.info('Fetching section content for:{topic}, {section_title}')       
        page = wikipedia.page(topic)
        content = page.section(section_title)
        if content:
            return {"content": content}
        else:
            return {"error": f"Section '{section_title}' not found in article '{topic}'."}
    except Exception as e:
        return {"error": str(e)}

# Run the MCP server
if __name__ == "__main__":
    print("Starting MCP Wikipedia Server...")
    mcp.run(transport="stdio")
