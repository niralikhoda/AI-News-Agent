import os
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from datetime import datetime, timedelta
import re
import asyncio
import logging
from getpass import getpass
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dateutil import parser as date_parser
import feedparser
import time
import schedule
from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Variable Loading ---
# Smartly load .env file from common locations
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("../.env"):
    load_dotenv("../.env")

# Fallback for API keys if not in .env
if not os.environ.get("NEWSAPI_KEY"):
    os.environ["NEWSAPI_KEY"] = ' Your news api ' # Example key, replace if needed
if not os.environ.get("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY not found in environment variables. The script will fail if it's required.")
if not os.environ.get("EMAIL_PASSWORD"):
    logger.warning("EMAIL_PASSWORD not found. Email notifications will be skipped.")
if not os.environ.get("EMAIL_ADDRESS"):
    logger.warning("EMAIL_ADDRESS not found. Email notifications will be skipped.")


# --- Model Initialization ---
try:
    model = "gpt-4o-mini"
    llm = ChatOpenAI(model=model, timeout=30, max_retries=3)
except Exception as e:
    logger.error(f"Fatal Error: Could not initialize OpenAI model: {e}")
    raise

# --- Custom Websites Configuration ---
CUSTOM_WEBSITES = [
    {
        "name": "MIT AI News",
        "url": "https://news.mit.edu/topic/artificial-intelligence2",
        "rss_feed": "https://news.mit.edu/rss/topic/artificial-intelligence2"
    },
    {
        "name": "Google AI Blog",
        "url": "https://blog.google/technology/ai/",
        "rss_feed": "https://blog.google/technology/ai/rss/"
    },
    {
        "name": "WSJ AI",
        "url": "https://www.wsj.com/tech/ai",
        "rss_feed": None  # WSJ doesn't have a public RSS for this section
    },
    {
        "name": "The Verge AI",
        "url": "https://www.theverge.com/ai-artificial-intelligence",
        "rss_feed": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"
    }
]

# --- State and Data Models ---
class GraphState(TypedDict):
    news_query: Annotated[str, "Input query to extract news search parameters from."]
    target_date: Annotated[str, "Target date for news (YYYY-MM-DD format)."]
    num_searches_remaining: Annotated[int, "Number of articles to search for."]
    newsapi_params: Annotated[dict, "Structured argument for the News API."]
    past_searches: Annotated[List[dict], "List of search params already used."]
    articles_metadata: Annotated[list[dict], "Article metadata response from the News API"]
    custom_articles: Annotated[List[dict], "Articles from custom websites."]
    scraped_urls: Annotated[List[str], "List of urls already scraped."]
    num_articles_tldr: Annotated[int, "Number of articles to create TL;DR for."]
    potential_articles: Annotated[List[dict], "Article with full text to consider summarizing."]
    tldr_articles: Annotated[List[dict], "Selected article TL;DRs."]
    formatted_results: Annotated[str, "Formatted results to display."]
    error_message: Annotated[str, "Error message if any step fails."]
    email_sent: Annotated[bool, "Whether email notification was sent."]

class NewsApiParams(BaseModel):
    q: str = Field(description="1-3 concise keyword search terms that are not too specific")
    sources: str = Field(description="comma-separated list of sources from: 'abc-news,abc-news-au,associated-press,australian-financial-review,axios,bbc-news,bbc-sport,bloomberg,business-insider,cbc-news,cbs-news,cnn,financial-post,fortune'", default="")
    from_date: str = Field(description="date in format 'YYYY-MM-DD' for target date", default="")
    to: str = Field(description="date in format 'YYYY-MM-DD' for target date", default="")
    language: str = Field(description="language of articles 'en' unless specified", default="en")
    sort_by: str = Field(description="sort by 'publishedAt' for recent news", default="publishedAt")

# --- Helper Functions ---
def is_same_day(article_date: str, target_date: str) -> bool:
    """Check if article date matches target date, robustly."""
    try:
        if not article_date or not target_date:
            return False
        parsed_date = date_parser.parse(article_date).date()
        target_parsed = datetime.strptime(target_date, "%Y-%m-%d").date()
        return parsed_date == target_parsed
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse date '{article_date}': {e}")
        return False

# --- Graph Nodes ---
def generate_newsapi_params(state: GraphState) -> GraphState:
    """Based on the query, generate News API params for the target date."""
    logger.info("Generating NewsAPI search parameters...")
    try:
        parser = JsonOutputParser(pydantic_object=NewsApiParams)
        target_date = state["target_date"]
        past_searches = state.get("past_searches", [])
        
        template = """
        You are an expert news search query generator. Your goal is to find relevant articles for a specific day.
        Create a parameter dictionary for the News API to find articles from {target_date} based on the user query: "{query}"

        You have already tried these searches, so create a NEW and DIFFERENT set of search terms:
        {past_searches}
        
        Follow these formatting instructions precisely:
        {format_instructions}

        IMPORTANT: Set both 'from_date' and 'to' fields to '{target_date}' to search only that day.
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "target_date", "past_searches"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | llm | parser
        result = chain.invoke({
            "query": state["news_query"],
            "target_date": target_date,
            "past_searches": past_searches,
        })
        result['from_date'] = target_date
        result['to'] = target_date
        state["newsapi_params"] = result
        logger.info(f"Generated NewsAPI params: {result}")
    except Exception as e:
        logger.error(f"Error in generate_newsapi_params: {e}")
        state["error_message"] = f"Failed to generate search parameters: {e}"
    return state

def retrieve_articles_metadata(state: GraphState) -> GraphState:
    """Fetch article metadata from NewsAPI and custom websites."""
    logger.info("Retrieving articles metadata...")
    try:
        # NewsAPI call
        newsapi_params = state["newsapi_params"]
        api_key = os.getenv('NEWSAPI_KEY')
        if not api_key:
            raise ValueError("NewsAPI key not found in environment variables.")
        
        newsapi = NewsApiClient(api_key=api_key)
        # NewsAPI client expects 'from_param' instead of 'from'
        api_params = newsapi_params.copy()
        if 'from_date' in api_params:
            api_params['from_param'] = api_params.pop('from_date')
        
        logger.info(f"Calling NewsAPI with: {api_params}")
        api_response = newsapi.get_everything(**api_params)
        
        # Filter and format NewsAPI articles
        target_date = state["target_date"]
        newsapi_articles = [
            article for article in api_response.get('articles', [])
            if is_same_day(article.get('publishedAt'), target_date)
        ]
        logger.info(f"Found {len(newsapi_articles)} articles from NewsAPI for {target_date}.")

        # Scrape custom websites
        custom_articles = scrape_custom_websites(target_date)
        
        # Combine and de-duplicate articles
        all_articles = newsapi_articles + custom_articles
        current_urls = state.get("scraped_urls", [])
        unique_new_articles = []
        for article in all_articles:
            if article.get('url') and article['url'] not in current_urls:
                unique_new_articles.append(article)
                current_urls.append(article['url'])

        state["articles_metadata"] = unique_new_articles
        state["scraped_urls"] = current_urls
        state["past_searches"] = state.get("past_searches", []) + [newsapi_params]
        state["num_searches_remaining"] -= 1
        logger.info(f"Retrieved {len(unique_new_articles)} new unique articles.")
    except Exception as e:
        logger.error(f"Error retrieving articles metadata: {e}")
        state["error_message"] = f"Failed to retrieve articles: {e}"
        state["articles_metadata"] = []
    return state

def scrape_custom_websites(target_date: str) -> List[dict]:
    """Scrape a predefined list of websites for AI news on a specific date."""
    logger.info(f"Scraping custom websites for date: {target_date}")
    custom_articles = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    for site in CUSTOM_WEBSITES:
        try:
            # Prioritize RSS feeds for reliability and efficiency
            if site['rss_feed']:
                feed = feedparser.parse(site['rss_feed'])
                for entry in feed.entries:
                    if is_same_day(entry.get('published'), target_date):
                        custom_articles.append({
                            'title': entry.get('title', 'N/A'),
                            'url': entry.get('link', ''),
                            'description': entry.get('summary', ''),
                            'published_at': entry.get('published'),
                            'source': site['name']
                        })
                continue # Move to next site after processing RSS
            
            # Fallback to HTML scraping if no RSS feed
            response = requests.get(site['url'], headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            # Add site-specific scraping logic here if needed
            # This is a generic example
            for article_tag in soup.find_all('article', limit=5):
                 title_tag = article_tag.find(['h1', 'h2', 'h3'])
                 link_tag = article_tag.find('a', href=True)
                 if title_tag and link_tag:
                     # Date extraction is complex and site-specific, omitted for this generic example
                     custom_articles.append({
                         'title': title_tag.get_text(strip=True),
                         'url': requests.compat.urljoin(site['url'], link_tag['href']),
                         'description': '',
                         'published_at': target_date, # Assume target date if not found
                         'source': site['name']
                     })
        except Exception as e:
            logger.warning(f"Could not scrape {site['name']}: {e}")
    
    logger.info(f"Found {len(custom_articles)} articles from custom websites.")
    return custom_articles

def retrieve_articles_text(state: GraphState) -> GraphState:
    """Scrape the full text content of articles from their URLs."""
    logger.info("Retrieving full text for articles...")
    potential_articles = state.get("potential_articles", [])
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    for article_meta in state.get("articles_metadata", []):
        url = article_meta.get('url')
        if not url:
            continue
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            # A simple heuristic to get the main content
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            if len(text) > 500: # Check for a minimum content length
                article_meta['text'] = text[:4000] # Limit text to avoid excessive token usage
                potential_articles.append(article_meta)
                logger.info(f"Successfully scraped: {url}")
            else:
                logger.warning(f"Content too short, skipping: {url}")
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")

    state["potential_articles"] = potential_articles
    return state

def select_top_urls(state: GraphState) -> GraphState:
    """Use an LLM to select the most relevant articles to summarize."""
    logger.info("Selecting top articles for summarization...")
    try:
        potential_articles = state["potential_articles"]
        if not potential_articles:
            state["tldr_articles"] = []
            return state

        article_previews = "\n---\n".join([
            f"URL: {a['url']}\nTitle: {a['title']}\nDescription: {a.get('description', '')}"
            for a in potential_articles
        ])
        
        prompt = f"""
        You are a news editor. Based on the user's query "{state['news_query']}", select up to {state['num_articles_tldr']} of the most relevant, significant, and unique articles from the list below.
        Prioritize major announcements, breakthroughs, and diverse topics. Avoid duplicate stories from different sources.

        Respond with ONLY the list of URLs, one URL per line. Do not add any other text or explanation.

        Available Articles:
        {article_previews}
        """
        response = llm.invoke(prompt)
        urls = response.content.strip().split('\n')
        
        selected_articles = [p for p in potential_articles if p['url'] in urls]
        state["tldr_articles"] = selected_articles
        logger.info(f"Selected {len(selected_articles)} articles for TL;DR.")
    except Exception as e:
        logger.error(f"Error selecting top URLs: {e}")
        # Fallback to selecting the first N articles if LLM fails
        state["tldr_articles"] = state.get("potential_articles", [])[:state["num_articles_tldr"]]
    return state

async def summarize_articles_parallel(state: GraphState) -> GraphState:
    """Summarize the selected articles in parallel."""
    logger.info("Summarizing articles...")
    tldr_articles = state.get("tldr_articles", [])
    if not tldr_articles:
        return state

    async def summarize_article(article):
        prompt = f"""
        Create a concise, bulleted summary of the following article. Focus on the key takeaways and significance.
        Format your response exactly as follows, with no extra text before or after:

        {article['title']}
        URL: {article['url']}
        Source: {article.get('source', 'N/A')} | Published: {date_parser.parse(article.get('publishedAt', 'now')).strftime('%B %d, %Y')}
        * [First key point: Main announcement or finding]
        * [Second key point: Impact or implications]
        * [Third key point: Important detail or future outlook]

        Article Text:
        {article.get('text', '')}
        """
        try:
            response = await llm.ainvoke(prompt)
            article['summary'] = response.content
            logger.info(f"Successfully summarized: {article['title']}")
        except Exception as e:
            logger.warning(f"Could not summarize '{article['title']}': {e}")
            article['summary'] = f"{article['title']}\nURL: {article['url']}\n* Summary failed to generate."
        return article

    summarized_articles = await asyncio.gather(*(summarize_article(a) for a in tldr_articles))
    state["tldr_articles"] = summarized_articles
    return state

def format_results(state: GraphState) -> GraphState:
    """Format the final summaries into a single string for display or email."""
    logger.info("Formatting final results...")
    target_date_str = datetime.strptime(state["target_date"], "%Y-%m-%d").strftime("%B %d, %Y")
    tldr_articles = state.get("tldr_articles", [])

    if not tldr_articles:
        digest = f"📰 AI News Digest for {target_date_str}\n\nNo relevant AI news found for today."
    else:
        summaries = [a.get('summary', 'Summary not available.') for a in tldr_articles]
        digest = f"📰 AI News Digest for {target_date_str}\n\n"
        digest += "\n\n---\n\n".join(summaries)
    
    state["formatted_results"] = digest
    return state

def send_email_notification(state: GraphState) -> GraphState:
    """Send the formatted news digest via email."""
    logger.info("Preparing to send email notification...")
    email_address = os.getenv("EMAIL_ADDRESS")
    email_password = os.getenv("EMAIL_PASSWORD")
    recipient_email = os.getenv("RECIPIENT_EMAIL", email_address)

    if not email_address or not email_password:
        logger.warning("Email credentials not found. Skipping email.")
        state["email_sent"] = False
        return state

    try:
        msg = MIMEMultipart()
        msg['From'] = email_address
        msg['To'] = recipient_email
        target_date_str = datetime.strptime(state["target_date"], "%Y-%m-%d").strftime("%B %d, %Y")
        msg['Subject'] = f"🤖 Your AI News Digest for {target_date_str}"
        msg.attach(MIMEText(state["formatted_results"], 'plain'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.send_message(msg)
            logger.info(f"Email sent successfully to {recipient_email}")
            state["email_sent"] = True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        state["email_sent"] = False
    return state

# --- Conditional Logic ---
def articles_text_decision(state: GraphState) -> Literal["generate_newsapi_params", "select_top_urls", "format_results"]:
    """Decide the next step after scraping article text."""
    if state.get("error_message"):
        return "format_results"
    
    potential_articles_count = len(state.get("potential_articles", []))
    if potential_articles_count < state["num_articles_tldr"] and state["num_searches_remaining"] > 0:
        logger.info("Not enough articles, searching again.")
        return "generate_newsapi_params"
    elif potential_articles_count == 0:
        logger.info("No potential articles found, finishing up.")
        return "format_results"
    else:
        logger.info("Sufficient articles found, proceeding to selection.")
        return "select_top_urls"

# --- Graph Definition ---
def build_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(GraphState)
    workflow.add_node("generate_newsapi_params", generate_newsapi_params)
    workflow.add_node("retrieve_articles_metadata", retrieve_articles_metadata)
    workflow.add_node("retrieve_articles_text", retrieve_articles_text)
    workflow.add_node("select_top_urls", select_top_urls)
    workflow.add_node("summarize_articles_parallel", summarize_articles_parallel)
    workflow.add_node("format_results", format_results)
    workflow.add_node("send_email_notification", send_email_notification)

    workflow.add_edge(START, "generate_newsapi_params")
    workflow.add_edge("generate_newsapi_params", "retrieve_articles_metadata")
    workflow.add_edge("retrieve_articles_metadata", "retrieve_articles_text")
    workflow.add_conditional_edges(
        "retrieve_articles_text",
        articles_text_decision,
        {
            "generate_newsapi_params": "generate_newsapi_params",
            "select_top_urls": "select_top_urls",
            "format_results": "format_results"
        }
    )
    workflow.add_edge("select_top_urls", "summarize_articles_parallel")
    workflow.add_edge("summarize_articles_parallel", "format_results")
    workflow.add_edge("format_results", "send_email_notification")
    workflow.add_edge("send_email_notification", END)
    
    return workflow.compile()

# --- Main Execution and CLI ---
async def run_digest(query: str, target_date_str: str, num_searches: int, num_articles: int):
    """Run the news digest workflow with specified parameters."""
    app = build_graph()
    initial_state = {
        "news_query": query,
        "target_date": target_date_str,
        "num_searches_remaining": num_searches,
        "num_articles_tldr": num_articles,
        "past_searches": [],
        "potential_articles": [],
        "scraped_urls": [],
    }
    try:
        final_state = await app.ainvoke(initial_state)
        print("\n--- FINAL DIGEST ---")
        print(final_state.get("formatted_results", "No digest generated."))
        print("\n--- WORKFLOW COMPLETE ---")
        print(f"Email Sent: {final_state.get('email_sent')}")
        if final_state.get('error_message'):
            print(f"Errors: {final_state.get('error_message')}")
    except Exception as e:
        logger.error(f"Critical error during workflow execution: {e}")

def run_scheduler():
    """Run the news digest on a daily schedule."""
    logger.info("Scheduler started. Digest will run daily at 08:00.")
    
    def job():
        logger.info("Executing scheduled daily digest job...")
        asyncio.run(run_digest(
            query="Latest news in AI, machine learning, and large language models",
            target_date_str=datetime.now().strftime("%Y-%m-%d"),
            num_searches=3,
            num_articles=5
        ))

    schedule.every().day.at("08:00").do(job)
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")

def test_email_connection():
    """Test SMTP connection and credentials."""
    logger.info("Testing email configuration...")
    # Implementation from your original script
    # ... (omitted for brevity, but can be pasted here) ...

def setup_email_config():
    """Provide instructions for setting up email credentials."""
    logger.info("Displaying email setup instructions...")
    # Implementation from your original script
    # ... (omitted for brevity, but can be pasted here) ...

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the AI News Digest agent.")
    parser.add_argument("command", nargs="?", default="run", help="Command to execute: run, schedule, test-email, setup")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--query", default="AI artificial intelligence news", help="The news topic to search for.")
    
    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(run_digest(query=args.query, target_date_str=args.date, num_searches=3, num_articles=5))
    elif args.command == "schedule":
        run_scheduler()
    elif args.command == "test-email":
        test_email_connection()
    elif args.command == "setup":
        setup_email_config()
    else:
        print(f"Unknown command: {args.command}")
