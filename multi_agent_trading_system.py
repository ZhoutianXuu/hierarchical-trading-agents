"""
Hierarchical Multi-Agent LLM Architecture with Local Ollama
Modified to use Qwen3:4b via local Ollama

Team 9: Zhoutian Xu, Raymond Tao, Jiashuo Xu
Course: SYSEN 5530
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

# Simplified imports - no heavy transformers needed
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    try:
        from langchain.vectorstores import FAISS
    except ImportError:
        FAISS = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None


@dataclass
class MarketData:
    """Container for market data inputs"""
    ticker: str
    date: datetime
    ohlc: Dict[str, float]  # open, high, low, close
    volume: float
    valuation_ratios: Dict[str, float]  # PE, PB, PS, PCF
    fundamentals: Dict[str, Any]
    news: List[str]
    prev_close: Optional[float] = None


@dataclass
class AgentOutput:
    """Standardized output from each agent"""
    agent_name: str
    analysis: str
    confidence: float
    key_signals: List[str]
    reasoning_trace: str
    timestamp: datetime


@dataclass
class TradingDecision:
    """Final trading decision output"""
    ticker: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    rationale: str
    supporting_agents: List[str]
    risk_assessment: Dict[str, Any]
    timestamp: datetime


class RAGKnowledgeBase:
    """Retrieval-Augmented Generation Knowledge Layer (Optional)"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", enabled: bool = False):
        self.enabled = enabled
        if not enabled:
            print("RAG is disabled - will skip retrieval steps")
            return
        
        # Check if required libraries are available
        if HuggingFaceEmbeddings is None or FAISS is None or RecursiveCharacterTextSplitter is None:
            print("⚠ Warning: RAG libraries not fully available. Install with:")
            print("  pip install langchain-community sentence-transformers faiss-cpu")
            self.enabled = False
            return
            
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
    def index_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Index financial documents, news, and company filings"""
        if not self.enabled:
            return
            
        texts = self.text_splitter.create_documents(
            documents,
            metadatas=metadata if metadata else [{}] * len(documents)
        )
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        
    def retrieve_relevant_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant documents for a given query"""
        if not self.enabled or self.vector_store is None:
            return []
        
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]


class OllamaClient:
    """Client for interacting with local Ollama API"""
    
    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        
        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✓ Connected to Ollama at {base_url}")
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if any(self.model in m for m in model_names):
                    print(f"✓ Model {self.model} is available")
                else:
                    print(f"⚠ Warning: Model {self.model} not found. Available models: {model_names}")
            else:
                print(f"⚠ Warning: Could not connect to Ollama (status {response.status_code})")
        except Exception as e:
            print(f"⚠ Warning: Could not connect to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512, no_think: bool = True) -> str:
        """
        Generate text using Ollama API
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            no_think: If True, disable reasoning/thinking process (recommended for Qwen3)
        """
        try:
            # Configure options
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            
            # For Qwen3 models, add specific parameters to disable thinking
            if no_think:
                # Add system instruction to prevent reasoning output
                system_msg = "Answer directly and concisely. Do not show your reasoning or thinking process. Provide only the final answer."
                
                # Modify prompt to discourage thinking
                modified_prompt = f"{prompt}\n\nProvide a direct, concise answer without showing your reasoning."
            else:
                system_msg = ""
                modified_prompt = prompt
            
            payload = {
                "model": self.model,
                "prompt": modified_prompt,
                "stream": False,
                "options": options,
            }
            
            # Add system message if no_think is enabled
            if system_msg:
                payload["system"] = system_msg
            
            response = requests.post(self.generate_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                return f"Error: Ollama returned status {response.status_code}"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"


class BaseAgent:
    """Base class for all specialized LLM agents using Ollama"""
    
    def __init__(
        self,
        agent_name: str,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434"
    ):
        self.agent_name = agent_name
        self.ollama = OllamaClient(model=model, base_url=base_url)
    
    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512, no_think: bool = True) -> str:
        """Generate response from Ollama"""
        return self.ollama.generate(prompt, temperature=temperature, max_tokens=max_tokens, no_think=no_think)
    
    def analyze(self, market_data: MarketData, retrieved_context: List[str]) -> AgentOutput:
        """To be implemented by specialized agents"""
        raise NotImplementedError


class FundamentalAgent(BaseAgent):
    """Agent specialized in fundamental analysis"""
    
    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434"):
        super().__init__("Fundamental_Agent", model, base_url)
    
    def analyze(self, market_data: MarketData, retrieved_context: List[str]) -> AgentOutput:
        """Perform fundamental analysis"""
        
        prev_close_str = f"${market_data.prev_close:.2f}" if market_data.prev_close else "N/A"
        
        prompt = f"""Analyze this stock data and provide ONLY the final analysis. Do not show your reasoning.

Ticker: {market_data.ticker}
Date: {market_data.date}

Price: ${market_data.ohlc.get('close', 0):.2f}
Previous Close: {prev_close_str}

Valuation Ratios:
- P/E Ratio: {market_data.valuation_ratios.get('pe', 'N/A')}
- P/B Ratio: {market_data.valuation_ratios.get('pb', 'N/A')}
- P/S Ratio: {market_data.valuation_ratios.get('ps', 'N/A')}
- P/CF Ratio: {market_data.valuation_ratios.get('pcf_ttm', 'N/A')}

Provide your analysis in 2-3 sentences:
1. Valuation assessment (overvalued/undervalued)
2. Key fundamental signals
3. Your confidence level

Analysis:"""

        response = self.generate_response(prompt, temperature=0.7, max_tokens=256)
        
        # Extract signals and confidence
        confidence = 0.7  # Default
        key_signals = self._extract_signals(response)
        
        return AgentOutput(
            agent_name=self.agent_name,
            analysis=response,
            confidence=confidence,
            key_signals=key_signals,
            reasoning_trace="Fundamental analysis based on valuation ratios",
            timestamp=datetime.now()
        )
    
    def _extract_signals(self, analysis: str) -> List[str]:
        """Extract key signals from analysis text"""
        signals = []
        analysis_lower = analysis.lower()
        
        if "undervalued" in analysis_lower:
            signals.append("UNDERVALUED")
        if "overvalued" in analysis_lower:
            signals.append("OVERVALUED")
        if "strong" in analysis_lower and ("financial" in analysis_lower or "fundamental" in analysis_lower):
            signals.append("STRONG_FUNDAMENTALS")
        if "weak" in analysis_lower:
            signals.append("WEAK_FUNDAMENTALS")
            
        return signals


class SentimentAgent(BaseAgent):
    """Agent specialized in sentiment and news analysis"""
    
    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434"):
        super().__init__("Sentiment_Agent", model, base_url)
    
    def analyze(self, market_data: MarketData, retrieved_context: List[str]) -> AgentOutput:
        """Perform sentiment analysis on news"""
        # Get first 3 news items
        news_preview = []
        for i, news_item in enumerate(market_data.news[:3]):
            if isinstance(news_item, dict):
                title = news_item.get('title', '')
                news_preview.append(f"{i+1}. {title}")
            else:
                news_preview.append(f"{i+1}. {str(news_item)[:100]}")
        
        news_str = "\n".join(news_preview) if news_preview else "No news available"
        
        prompt = f"""Analyze news sentiment. Provide ONLY the final analysis without showing reasoning.

Ticker: {market_data.ticker}
Date: {market_data.date}

Recent News:
{news_str}

Provide your analysis in 2-3 sentences:
1. Overall sentiment (positive/negative/neutral)
2. Key themes
3. Market impact

Analysis:"""

        response = self.generate_response(prompt, temperature=0.7, max_tokens=256)
        
        confidence = 0.75
        key_signals = self._extract_sentiment_signals(response)
        
        return AgentOutput(
            agent_name=self.agent_name,
            analysis=response,
            confidence=confidence,
            key_signals=key_signals,
            reasoning_trace=f"Sentiment analysis of {len(market_data.news)} news items",
            timestamp=datetime.now()
        )
    
    def _extract_sentiment_signals(self, analysis: str) -> List[str]:
        """Extract sentiment signals"""
        signals = []
        analysis_lower = analysis.lower()
        
        if "positive" in analysis_lower:
            signals.append("POSITIVE_SENTIMENT")
        if "negative" in analysis_lower:
            signals.append("NEGATIVE_SENTIMENT")
        if "bullish" in analysis_lower:
            signals.append("BULLISH")
        if "bearish" in analysis_lower:
            signals.append("BEARISH")
            
        return signals


class TechnicalAgent(BaseAgent):
    """Agent specialized in technical analysis"""
    
    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434"):
        super().__init__("Technical_Agent", model, base_url)
    
    def analyze(self, market_data: MarketData, retrieved_context: List[str]) -> AgentOutput:
        """Perform technical analysis"""
        
        # Calculate basic metrics
        price_change = 0
        prev_close_str = "N/A"
        if market_data.prev_close and market_data.prev_close > 0:
            price_change = ((market_data.ohlc['close'] - market_data.prev_close) / market_data.prev_close) * 100
            prev_close_str = f"${market_data.prev_close:.2f}"
        
        prompt = f"""Analyze price action. Provide ONLY the final analysis without showing reasoning.

Ticker: {market_data.ticker}
Date: {market_data.date}

Price Data:
- Open: ${market_data.ohlc.get('open', 0):.2f}
- High: ${market_data.ohlc.get('high', 0):.2f}
- Low: ${market_data.ohlc.get('low', 0):.2f}
- Close: ${market_data.ohlc.get('close', 0):.2f}
- Previous Close: {prev_close_str}
- Change: {price_change:.2f}%

Volume: {market_data.volume:,.0f}

Provide your analysis in 2-3 sentences:
1. Price momentum
2. Volume analysis
3. Key technical signals

Analysis:"""

        response = self.generate_response(prompt, temperature=0.7, max_tokens=256)
        
        confidence = 0.65
        key_signals = self._extract_technical_signals(response, price_change)
        
        return AgentOutput(
            agent_name=self.agent_name,
            analysis=response,
            confidence=confidence,
            key_signals=key_signals,
            reasoning_trace="Technical analysis of price and volume",
            timestamp=datetime.now()
        )
    
    def _extract_technical_signals(self, analysis: str, price_change: float) -> List[str]:
        """Extract technical signals"""
        signals = []
        analysis_lower = analysis.lower()
        
        if "breakout" in analysis_lower:
            signals.append("BREAKOUT")
        if "support" in analysis_lower:
            signals.append("AT_SUPPORT")
        if "resistance" in analysis_lower:
            signals.append("AT_RESISTANCE")
        if price_change > 2:
            signals.append("STRONG_UPWARD_MOMENTUM")
        elif price_change < -2:
            signals.append("STRONG_DOWNWARD_MOMENTUM")
            
        return signals


class RiskManagementAgent(BaseAgent):
    """Agent specialized in risk assessment"""
    
    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434"):
        super().__init__("Risk_Management_Agent", model, base_url)
    
    def analyze(self, market_data: MarketData, retrieved_context: List[str]) -> AgentOutput:
        """Perform risk analysis"""
        
        price_change = 0
        if market_data.prev_close and market_data.prev_close > 0:
            price_change = abs((market_data.ohlc['close'] - market_data.prev_close) / market_data.prev_close) * 100
        
        prompt = f"""Assess trading risk. Provide ONLY the final assessment without showing reasoning.

Ticker: {market_data.ticker}
Current Price: ${market_data.ohlc['close']:.2f}
Daily Change: {price_change:.2f}%
Volume: {market_data.volume:,.0f}

Provide your risk assessment in 2-3 sentences:
1. Overall risk level (low/medium/high)
2. Key risk factors
3. Risk mitigation

Assessment:"""

        response = self.generate_response(prompt, temperature=0.8, max_tokens=256)
        
        confidence = 0.8
        key_signals = self._extract_risk_signals(response)
        
        return AgentOutput(
            agent_name=self.agent_name,
            analysis=response,
            confidence=confidence,
            key_signals=key_signals,
            reasoning_trace="Risk assessment based on market conditions",
            timestamp=datetime.now()
        )
    
    def _extract_risk_signals(self, analysis: str) -> List[str]:
        """Extract risk signals"""
        signals = []
        analysis_lower = analysis.lower()
        
        if "high risk" in analysis_lower:
            signals.append("HIGH_RISK")
        if "low risk" in analysis_lower:
            signals.append("LOW_RISK")
        if "medium risk" in analysis_lower:
            signals.append("MEDIUM_RISK")
        if "volatile" in analysis_lower or "volatility" in analysis_lower:
            signals.append("HIGH_VOLATILITY")
            
        return signals


class ReviewerAgent(BaseAgent):
    """Reviewer Agent that synthesizes intermediate reasoning"""
    
    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434"):
        super().__init__("Reviewer_Agent", model, base_url)
    
    def synthesize(self, agent_outputs: List[AgentOutput]) -> str:
        """Synthesize analyses from multiple agents"""
        
        # Compile summaries
        summaries = []
        for output in agent_outputs:
            summary = f"{output.agent_name}: {output.analysis[:150]}... (Confidence: {output.confidence:.2f}, Signals: {', '.join(output.key_signals[:2])})"
            summaries.append(summary)
        
        analyses_text = "\n\n".join(summaries)
        
        prompt = f"""Synthesize these expert analyses. Provide ONLY the final synthesis without showing reasoning.

{analyses_text}

Provide your synthesis in 3-4 sentences:
1. Key consensus points
2. Main conflicts
3. Overall outlook
4. Recommended direction

Synthesis:"""

        synthesis = self.generate_response(prompt, temperature=0.6, max_tokens=512)
        return synthesis


class DecisionAgent(BaseAgent):
    """Decision Agent that produces final trading signals"""
    
    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://localhost:11434"):
        super().__init__("Decision_Agent", model, base_url)
    
    def make_decision(
        self,
        market_data: MarketData,
        agent_outputs: List[AgentOutput],
        synthesis: str
    ) -> TradingDecision:
        """Generate final trading decision"""
        
        # Compile all signals
        all_signals = []
        for output in agent_outputs:
            all_signals.extend(output.key_signals)
        
        signals_summary = ", ".join(set(all_signals)) if all_signals else "No clear signals"
        
        prompt = f"""Make final trading decision. Provide ONLY the decision without showing reasoning process.

Ticker: {market_data.ticker}
Current Price: ${market_data.ohlc['close']:.2f}

Synthesis: {synthesis}
Signals: {signals_summary}

Respond ONLY with:
DECISION: [BUY/SELL/HOLD]
CONFIDENCE: [0.0-1.0]
RATIONALE: [2-3 sentences]

Response:"""

        response = self.generate_response(prompt, temperature=0.5, max_tokens=512)
        
        # Parse decision
        action = self._extract_action(response)
        confidence = self._extract_confidence(response)
        
        # Risk assessment
        risk_agent_output = next(
            (out for out in agent_outputs if out.agent_name == "Risk_Management_Agent"),
            None
        )
        risk_assessment = {
            "level": "MEDIUM",
            "factors": risk_agent_output.key_signals if risk_agent_output else []
        }
        
        supporting_agents = [out.agent_name for out in agent_outputs if out.confidence > 0.6]
        
        return TradingDecision(
            ticker=market_data.ticker,
            action=action,
            confidence=confidence,
            rationale=response,
            supporting_agents=supporting_agents,
            risk_assessment=risk_assessment,
            timestamp=datetime.now()
        )
    
    def _extract_action(self, response: str) -> str:
        """Extract trading action from response"""
        response_upper = response.upper()
        if "DECISION: BUY" in response_upper or "BUY" in response_upper[:100]:
            return "BUY"
        elif "DECISION: SELL" in response_upper or "SELL" in response_upper[:100]:
            return "SELL"
        else:
            return "HOLD"
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        import re
        match = re.search(r'CONFIDENCE:\s*(\d*\.?\d+)', response, re.IGNORECASE)
        if match:
            try:
                conf = float(match.group(1))
                # If it looks like a percentage, convert
                if conf > 1:
                    conf = conf / 100
                return max(0.0, min(1.0, conf))
            except:
                pass
        return 0.5


class HierarchicalMultiAgentSystem:
    """Main orchestrator for the hierarchical multi-agent system using local Ollama"""
    
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        use_rag: bool = False,
        no_think: bool = True  # Disable reasoning/thinking output for faster, more direct responses
    ):
        print(f"\n{'='*60}")
        print("Initializing Hierarchical Multi-Agent System")
        print(f"Model: {model} (Local Ollama)")
        print(f"RAG: {'Enabled' if use_rag else 'Disabled'}")
        print(f"No-Think Mode: {'Enabled' if no_think else 'Disabled'}")
        print(f"{'='*60}\n")
        
        self.no_think = no_think
        
        # Initialize RAG Knowledge Base (optional)
        self.rag_kb = RAGKnowledgeBase(enabled=use_rag)
        
        # Initialize specialized agents (Layer 1)
        print("Initializing agents...")
        self.fundamental_agent = FundamentalAgent(model, base_url)
        self.sentiment_agent = SentimentAgent(model, base_url)
        self.technical_agent = TechnicalAgent(model, base_url)
        self.risk_agent = RiskManagementAgent(model, base_url)
        
        # Initialize coordination layer (Layer 2)
        self.reviewer_agent = ReviewerAgent(model, base_url)
        self.decision_agent = DecisionAgent(model, base_url)
        
        print("✓ Multi-Agent System initialized!\n")
    
    def process_trading_decision(self, market_data: MarketData) -> TradingDecision:
        """Process a complete trading decision through the hierarchical system"""
        
        print(f"\n{'='*60}")
        print(f"Processing: {market_data.ticker} on {market_data.date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}\n")
        
        # Step 1: RAG retrieval (optional)
        retrieved_context = []
        if self.rag_kb.enabled:
            print("Step 1: Retrieving relevant context...")
            query = f"{market_data.ticker} stock analysis"
            retrieved_context = self.rag_kb.retrieve_relevant_context(query, k=3)
            print(f"Retrieved {len(retrieved_context)} documents\n")
        
        # Step 2: Parallel analysis by specialized agents
        print("Step 2: Running agent analyses...")
        agent_outputs = []
        
        print("  - Fundamental Agent...")
        agent_outputs.append(self.fundamental_agent.analyze(market_data, retrieved_context))
        
        print("  - Sentiment Agent...")
        agent_outputs.append(self.sentiment_agent.analyze(market_data, retrieved_context))
        
        print("  - Technical Agent...")
        agent_outputs.append(self.technical_agent.analyze(market_data, retrieved_context))
        
        print("  - Risk Management Agent...")
        agent_outputs.append(self.risk_agent.analyze(market_data, retrieved_context))
        print()
        
        # Step 3: Synthesis by Reviewer Agent
        print("Step 3: Synthesizing analyses...")
        synthesis = self.reviewer_agent.synthesize(agent_outputs)
        print()
        
        # Step 4: Final decision
        print("Step 4: Making final decision...")
        decision = self.decision_agent.make_decision(market_data, agent_outputs, synthesis)
        print()
        
        print(f"{'='*60}")
        print(f"DECISION: {decision.action} (Confidence: {decision.confidence:.2%})")
        print(f"{'='*60}\n")
        
        return decision
    
    def index_knowledge_base(self, documents: List[str], metadata: List[Dict] = None):
        """Index documents into the RAG knowledge base"""
        if self.rag_kb.enabled:
            self.rag_kb.index_documents(documents, metadata)


if __name__ == "__main__":
    print("Hierarchical Multi-Agent Trading System")
    print("Using Local Ollama Qwen3:4b")
    print("="*60)