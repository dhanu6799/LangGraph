# 🚀 # 🚀 LaunchLens – AI-Powered Product Launch Intelligence– LangGraph Sidekick

⚠️ **Status: In Development**  
This project is an experimental prototype and is still being actively developed.  
Features, prompts, and tool integrations may change frequently.

---

An AI-powered multi-agent system built with **LangGraph** and **Gradio** for analyzing competitor product launches.  
The assistant can perform three types of analysis:  

- 🔍 **Competitor Analysis** – positioning, strengths, weaknesses, strategic takeaways  
- 💬 **Market Sentiment** – positive/negative signals from social media, reviews, forums  
- 📈 **Launch Metrics** – KPIs, adoption, press coverage, traction insights  

It uses **OpenAI GPT models** with a tool-augmented workflow (Playwright browser, web search, Firecrawl, etc.) to provide fresh, evidence-backed reports.

---

## 📂 Project Structure

product_intel/
├── agents.py # Analyst role prompts (competitor, sentiment, metrics)

├── tools.py # External tools: Playwright, Serper, Wikipedia, Firecrawl, Python REPL

├── graph.py # LangGraph Sidekick orchestration (worker, tools, evaluator loop)

├── ui.py # Gradio UI (chatbot + buttons for analysis modes)

└── app1.py # Entry point to launch the app

---

## 🔑 Features

- **Multi-role prompts** (Competitor / Sentiment / Metrics)  
- **Tool-augmented reasoning** with:
  - Playwright (async browsing & scraping)
  - Google Serper (web search)
  - Wikipedia
  - Python REPL
  - File sandbox
  - Push notifications (Pushover)
  - Firecrawl (optional deep crawl/search)  
- **Evaluator loop** – ensures output meets success criteria, or asks clarifying questions  
- **Gradio interface** with persistent chat history and one-click mode selection  

---

## ⚙️ Setup

### 1. Clone repo & install dependencies
```bash
git clone https://github.com/yourusername/product-intel-sidekick.git
cd product-intel-sidekick
uv sync   # or pip install -r requirements.txt

2. Install Playwright browsers
playwright install

3. Configure environment variables

Create a .env file in the project root:

OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key   # optional, for Google search
FIRECRAWL_API_KEY=your_firecrawl_key # optional, for Firecrawl
PUSHOVER_TOKEN=your_pushover_token   # optional, for notifications
PUSHOVER_USER=your_pushover_user     # optional, for notifications

▶️ Run the app
uv run product_intel/app1.py

🖥️ Usage

Enter a company name or product in the chat.

Choose an analysis type:

🔍 Competitor Analysis

💬 Market Sentiment

📈 Launch Metrics

The assistant will gather evidence (via search/crawl) and generate a structured report.

You can refine the query or ask follow-up questions directly in chat.

📦 Requirements

Core libraries:

langgraph

langchain

langchain-openai

gradio

playwright

python-dotenv

requests

🛠️ Development Notes

Modular design: prompts (agents.py), tools (tools.py), orchestration (graph.py), UI (ui.py)

Easy to extend: add new tools in tools.py or new analyst roles in agents.py

Compatible with both Gradio (default) and Streamlit (if you want to port later)
