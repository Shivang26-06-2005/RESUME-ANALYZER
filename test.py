def _initialize_embeddings(self):
        # Available embedding models in Ollama (verified from ollama.com)
        embedding_models = [
            "nomic-embed-text",      # High-performing open embedding model 
            "mxbai-embed-large",     # Large multilingual embedding model
            "granite-embedding",      # IBM Granite embedding model
            "snowflake-arctic-embed" # Snowflake's embedding model
        ]
        
        print("Setting up embedding models...")
        # Resume-JD Semantic Matching System - FIXED VERSION
# Architecture: Gemma3 1B (Text Generation) + Working Embeddings + Chroma Vector Store
# Tech Stack: Python, LangChain, LangGraph, Ollama, HuggingFace, spaCy
# 100% FREE - No API costs!

import os
import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import re
from datetime import datetime

# Core libraries
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymupdf  # fitz
import docx2txt
from fuzzywuzzy import fuzz

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Pydantic models
from pydantic import BaseModel

# Sentence Transformers as reliable fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
    raise

@dataclass
class MatchingResult:
    relevance_score: float
    missing_skills: List[str]
    missing_projects: List[str]
    missing_certifications: List[str]
    verdict: str
    suggestions: List[str]
    semantic_similarity: float
    keyword_similarity: float

class ParsedResume(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    skills: List[str] = []
    experience: List[str] = []
    education: List[str] = []
    projects: List[str] = []
    certifications: List[str] = []
    raw_text: str = ""

class ParsedJD(BaseModel):
    role_title: str = ""
    must_have_skills: List[str] = []
    good_to_have_skills: List[str] = []
    qualifications: List[str] = []
    experience_required: str = ""
    raw_text: str = ""

class MatchingState(TypedDict):
    resume: ParsedResume
    jd: ParsedJD
    semantic_score: float
    final_score: float
    missing_items: Dict[str, List[str]]
    suggestions: List[str]
    verdict: str
    stage: str

class DocumentParser:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = pymupdf.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        try:
            text = docx2txt.process(docx_path)
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:()\-@]', '', text)
        return text.strip()

class ResumeParser:
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=ParsedResume)
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Extract structured information from resume text. Return ONLY valid JSON:
            {
                "name": "Full Name",
                "email": "email@example.com", 
                "phone": "phone number",
                "skills": ["skill1", "skill2"],
                "experience": ["Job Title at Company (Year-Year): Description"],
                "education": ["Degree, University (Year)"],
                "projects": ["Project Name: Description"],
                "certifications": ["Certification Name"]
            }
            {format_instructions}"""),
            HumanMessage(content="Resume: {resume_text}")
        ])
        self.chain = self.prompt | self.llm
    
    def parse_resume(self, resume_text: str) -> ParsedResume:
        try:
            doc = nlp(resume_text)
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
            phones = re.findall(r'\b[\+]?[1-9]?[0-9]{2,3}[-.\s]?[(]?[0-9]{1,4}[)]?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}\b', resume_text)
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON" and len(ent.text.split()) <= 3]

            response = self.chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "resume_text": resume_text[:3000]
            })
            
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed_data = json.loads(json_str)
                return ParsedResume(**parsed_data, raw_text=resume_text)
            else:
                raise ValueError("No JSON found")
                
        except Exception as e:
            logger.warning(f"Parse error: {e}, using fallback")
            return self._fallback_parse(resume_text, emails, phones, persons)
    
    def _fallback_parse(self, text: str, emails: List[str], phones: List[str], persons: List[str]) -> ParsedResume:
        # Enhanced skill extraction
        tech_skills = re.findall(r'\b(?:python|java|javascript|react|node|sql|aws|docker|kubernetes|git)\b', text, re.IGNORECASE)
        soft_skills = re.findall(r'\b(?:leadership|communication|teamwork|management|analysis)\b', text, re.IGNORECASE)
        skills = list(set([s.lower() for s in tech_skills + soft_skills]))
        
        return ParsedResume(
            name=persons[0] if persons else "",
            email=emails[0] if emails else "",
            phone=phones[0] if phones else "",
            skills=skills,
            raw_text=text
        )

class JDParser:
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=ParsedJD)
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Extract job requirements. Return ONLY valid JSON:
            {
                "role_title": "Job Title",
                "must_have_skills": ["required skill 1", "required skill 2"],
                "good_to_have_skills": ["nice to have 1", "nice to have 2"],
                "qualifications": ["degree requirement", "experience requirement"],
                "experience_required": "X years in Y field"
            }
            {format_instructions}"""),
            HumanMessage(content="Job Description: {jd_text}")
        ])
        self.chain = self.prompt | self.llm
    
    def parse_jd(self, jd_text: str) -> ParsedJD:
        try:
            response = self.chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "jd_text": jd_text[:3000]
            })
            
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed_data = json.loads(json_str)
                return ParsedJD(**parsed_data, raw_text=jd_text)
            else:
                raise ValueError("No JSON found")
                
        except Exception as e:
            logger.warning(f"JD parse error: {e}")
            return ParsedJD(raw_text=jd_text)

class SemanticMatcher:
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        self.embeddings = None
        self.sentence_model = None
        self._initialize_embeddings()
        
        self.feedback_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Analyze resume-job match and provide JSON feedback:
            {
                "missing_skills": ["Specific skill gaps with technical details"],
                "missing_projects": ["Project suggestions with technical implementation"],
                "missing_certifications": ["Relevant certifications with reasoning"],
                "suggestions": ["Actionable improvement recommendations"]
            }"""),
            HumanMessage(content="""Job: {role_title}
            Required: {must_have_skills}
            Preferred: {good_to_have_skills}
            Resume Skills: {resume_skills}
            Resume Projects: {resume_projects}
            Experience: {resume_experience}""")
        ])
        self.feedback_chain = self.feedback_prompt | self.llm
    
    def _initialize_embeddings(self):
        # Try multiple embedding approaches in order of preference
        
        # First, try to install required models automatically
        import subprocess
        
        models_to_install = [
            "nomic-embed-text",
            "mxbai-embed-large"
        ]
        
        print("Checking and installing embedding models...")
        for model in models_to_install:
            try:
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
                if model not in result.stdout:
                    print(f"Installing {model}...")
                    install_result = subprocess.run(['ollama', 'pull', model], capture_output=True, text=True, timeout=300)
                    if install_result.returncode == 0:
                        print(f"Successfully installed {model}")
                    else:
                        print(f"Failed to install {model}: {install_result.stderr}")
                else:
                    print(f"{model} already installed")
            except Exception as e:
                print(f"Could not check/install {model}: {e}")
        
        embedding_configs = [
            ("nomic-embed-text", "Nomic embedding model"),
            ("mxbai-embed-large", "MxBai large embedding model"), 
            ("all-minilm", "All-MiniLM embedding model")
        ]
        
        # Try Ollama embeddings first
        for model_name, description in embedding_configs:
            try:
                logger.info(f"Testing {description}: {model_name}")
                embeddings = OllamaEmbeddings(model=model_name, base_url="http://localhost:11434")
                
                # Test with short text
                test_embedding = embeddings.embed_query("test")
                if test_embedding and len(test_embedding) > 0:
                    self.embeddings = embeddings
                    logger.info(f"SUCCESS: Using {model_name} (dimension: {len(test_embedding)})")
                    return
            except Exception as e:
                logger.warning(f"Failed {model_name}: {e}")
                continue
        
        # Fallback to sentence-transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Trying sentence-transformers fallback...")
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                test_emb = self.sentence_model.encode(["test"])
                if test_emb is not None and len(test_emb) > 0:
                    logger.info(f"SUCCESS: Using sentence-transformers (dimension: {len(test_emb[0])})")
                    return
            except Exception as e:
                logger.warning(f"sentence-transformers failed: {e}")
        
        logger.error("All embedding methods failed - using TF-IDF fallback")
    
    def calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        resume_text = resume_text[:1500]
        jd_text = jd_text[:1500]
        
        try:
            # Try Ollama embeddings
            if self.embeddings:
                resume_emb = self.embeddings.embed_query(resume_text)
                jd_emb = self.embeddings.embed_query(jd_text)
                
                if resume_emb and jd_emb:
                    similarity = cosine_similarity(
                        np.array(resume_emb).reshape(1, -1),
                        np.array(jd_emb).reshape(1, -1)
                    )
                    normalized = (float(similarity[0][0]) + 1) / 2
                    logger.info(f"Ollama semantic similarity: {normalized:.3f}")
                    return round(normalized, 3)
            
            # Try sentence-transformers
            elif self.sentence_model:
                resume_emb = self.sentence_model.encode([resume_text])
                jd_emb = self.sentence_model.encode([jd_text])
                
                similarity = cosine_similarity(resume_emb, jd_emb)
                normalized = (float(similarity[0][0]) + 1) / 2
                logger.info(f"Sentence-transformers similarity: {normalized:.3f}")
                return round(normalized, 3)
                
        except Exception as e:
            logger.error(f"Embedding similarity failed: {e}")
        
        # TF-IDF fallback
        return self._tfidf_similarity(resume_text, jd_text)
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        try:
            tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words='english', max_features=1000)
            vectors = tfidf.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])
            score = max(0.0, min(1.0, float(similarity[0][0])))
            logger.info(f"TF-IDF similarity: {score:.3f}")
            return round(score, 3)
        except Exception as e:
            logger.error(f"TF-IDF similarity failed: {e}")
            return 0.0
    
    def generate_feedback_and_suggestions(self, resume: ParsedResume, jd: ParsedJD, score: float) -> Tuple[Dict[str, List[str]], List[str]]:
        try:
            response = self.feedback_chain.invoke({
                "role_title": jd.role_title,
                "must_have_skills": ', '.join(jd.must_have_skills),
                "good_to_have_skills": ', '.join(jd.good_to_have_skills),
                "resume_skills": ', '.join(resume.skills),
                "resume_projects": ', '.join(resume.projects),
                "resume_experience": ', '.join(resume.experience)
            })
            
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                feedback_data = json.loads(json_str)
                
                missing_items = {
                    "skills": feedback_data.get("missing_skills", []),
                    "projects": feedback_data.get("missing_projects", []),
                    "certifications": feedback_data.get("missing_certifications", [])
                }
                suggestions = feedback_data.get("suggestions", [])
                return missing_items, suggestions
            
        except Exception as e:
            logger.warning(f"Feedback generation failed: {e}")
        
        return self._basic_feedback(resume, jd)
    
    def _basic_feedback(self, resume: ParsedResume, jd: ParsedJD) -> Tuple[Dict[str, List[str]], List[str]]:
        missing_skills = []
        for skill in jd.must_have_skills:
            if not any(fuzz.ratio(skill.lower(), res_skill.lower()) > 70 for res_skill in resume.skills):
                missing_skills.append(f"Missing required skill: {skill}")
        
        missing_items = {
            "skills": missing_skills[:5],
            "projects": [f"Add projects demonstrating {jd.role_title} capabilities"] if not resume.projects else [],
            "certifications": [f"Consider certifications relevant to {jd.role_title}"]
        }
        
        suggestions = [
            "Enhance technical skills alignment with job requirements",
            "Add quantifiable achievements to experience descriptions",
            "Include relevant projects showcasing required competencies"
        ]
        
        return missing_items, suggestions

class ResumeJDMatcher:
    def __init__(self, model_name: str = "gemma3:1b"):
        self.llm = OllamaLLM(model=model_name, temperature=0.1)
        self.document_parser = DocumentParser()
        self.resume_parser = ResumeParser(self.llm)
        self.jd_parser = JDParser(self.llm)
        self.semantic_matcher = SemanticMatcher(self.llm)
        self.workflow = self._build_workflow()
        self.vectorstore = self._setup_vectorstore()
    
    def _setup_vectorstore(self):
        try:
            persist_directory = "./chroma_db"
            os.makedirs(persist_directory, exist_ok=True)
            
            if self.semantic_matcher.embeddings:
                return Chroma(
                    embedding_function=self.semantic_matcher.embeddings,
                    persist_directory=persist_directory,
                    collection_name="resume_matches"
                )
            elif self.semantic_matcher.sentence_model:
                from langchain_community.embeddings import SentenceTransformerEmbeddings
                embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
                return Chroma(
                    embedding_function=embeddings,
                    persist_directory=persist_directory,
                    collection_name="resume_matches"
                )
            else:
                logger.warning("No vectorstore - no embeddings available")
                return None
        except Exception as e:
            logger.error(f"Vectorstore setup failed: {e}")
            return None
    
    def _build_workflow(self):
        workflow = StateGraph(MatchingState)
        workflow.add_node("semantic_matching", self._semantic_node)
        workflow.add_node("scoring", self._scoring_node)
        workflow.add_node("feedback", self._feedback_node)
        workflow.add_node("verdict", self._verdict_node)
        workflow.add_edge("semantic_matching", "scoring")
        workflow.add_edge("scoring", "feedback")
        workflow.add_edge("feedback", "verdict")
        workflow.add_edge("verdict", END)
        workflow.set_entry_point("semantic_matching")
        return workflow.compile()
    
    def _semantic_node(self, state: MatchingState) -> MatchingState:
        semantic_score = self.semantic_matcher.calculate_semantic_similarity(
            state["resume"].raw_text, state["jd"].raw_text
        )
        state["semantic_score"] = semantic_score
        state["stage"] = "semantic_complete"
        return state
    
    def _scoring_node(self, state: MatchingState) -> MatchingState:
        state["final_score"] = min(round(state["semantic_score"] * 100, 2), 100)
        state["stage"] = "scoring_complete"
        return state
    
    def _feedback_node(self, state: MatchingState) -> MatchingState:
        missing_items, suggestions = self.semantic_matcher.generate_feedback_and_suggestions(
            state["resume"], state["jd"], state["final_score"]
        )
        state["missing_items"] = missing_items
        state["suggestions"] = suggestions
        state["stage"] = "feedback_complete"
        return state
    
    def _verdict_node(self, state: MatchingState) -> MatchingState:
        score = state["final_score"]
        if score >= 75:
            verdict = "High"
        elif score >= 50:
            verdict = "Medium"
        else:
            verdict = "Low"
        state["verdict"] = verdict
        state["stage"] = "complete"
        return state
    
    def match_resume_to_jd(self, resume_path: str, jd_path: str) -> MatchingResult:
        # Extract text
        if resume_path.lower().endswith('.pdf'):
            resume_text = self.document_parser.extract_text_from_pdf(resume_path)
        else:
            resume_text = self.document_parser.extract_text_from_docx(resume_path)
        
        if jd_path.lower().endswith('.pdf'):
            jd_text = self.document_parser.extract_text_from_pdf(jd_path)
        else:
            jd_text = self.document_parser.extract_text_from_docx(jd_path)
        
        if not resume_text or not jd_text:
            raise ValueError("Could not extract text from files")
        
        # Parse documents
        parsed_resume = self.resume_parser.parse_resume(resume_text)
        parsed_jd = self.jd_parser.parse_jd(jd_text)
        
        # Initialize workflow state
        initial_state = MatchingState(
            resume=parsed_resume,
            jd=parsed_jd,
            semantic_score=0.0,
            final_score=0.0,
            missing_items={},
            suggestions=[],
            verdict="",
            stage="initialized"
        )
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Store in vectorstore
        self._store_result(parsed_resume, parsed_jd, final_state)
        
        # Return result
        return MatchingResult(
            relevance_score=final_state['final_score'],
            missing_skills=final_state['missing_items'].get("skills", []),
            missing_projects=final_state['missing_items'].get("projects", []),
            missing_certifications=final_state['missing_items'].get("certifications", []),
            verdict=final_state['verdict'],
            suggestions=final_state['suggestions'],
            semantic_similarity=final_state['semantic_score'],
            keyword_similarity=0.0  # Not used in this version
        )
    
    def _store_result(self, resume: ParsedResume, jd: ParsedJD, final_state: MatchingState):
        try:
            if self.vectorstore:
                doc_text = f"Resume: {resume.name} | Skills: {', '.join(resume.skills)} | Job: {jd.role_title}"
                metadata = {
                    "resume_name": resume.name or "Unknown",
                    "job_title": jd.role_title,
                    "score": final_state['final_score'],
                    "verdict": final_state['verdict'],
                    "timestamp": datetime.now().isoformat()
                }
                
                self.vectorstore.add_texts(
                    texts=[doc_text],
                    metadatas=[metadata],
                    ids=[f"match_{datetime.now().strftime('%Y%m%d_%H%M%S')}"]
                )
                logger.info("Result stored in vectorstore")
        except Exception as e:
            logger.warning(f"Could not store result: {e}")

def main():
    """Main function with proper error handling"""
    print("\n=== Resume-JD Matching System ===")
    
    # Check if Ollama is running
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("Ollama is not running or not installed.")
            print("Please install Ollama from https://ollama.ai and run 'ollama serve'")
            return
    except Exception:
        print("Cannot connect to Ollama. Please ensure it's installed and running.")
        print("1. Install from: https://ollama.ai")
        print("2. Run: ollama serve")
        return
    
    try:
        print("Initializing system...")
        matcher = ResumeJDMatcher()
        
        # Test files
        resume_file = "14585273.pdf"
        jd_file = "sample_jd_2.pdf"
        
        if not os.path.exists(resume_file) or not os.path.exists(jd_file):
            print(f"Files not found. Please ensure {resume_file} and {jd_file} exist.")
            return
        
        print(f"Processing: {resume_file} vs {jd_file}")
        result = matcher.match_resume_to_jd(resume_file, jd_file)
        
        print(f"\n=== RESULTS ===")
        print(f"Overall Score: {result.relevance_score:.1f}/100")
        print(f"Match Level: {result.verdict}")
        print(f"Semantic Similarity: {result.semantic_similarity:.3f}")
        
        print(f"\nMissing Skills:")
        for skill in result.missing_skills[:3]:
            print(f"  • {skill}")
        
        print(f"\nSuggestions:")
        for suggestion in result.suggestions[:3]:
            print(f"  • {suggestion}")
        
        print(f"\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Install required models:")
        print("   ollama pull gemma3:1b")
        print("   ollama pull nomic-embed-text")
        print("3. Install Python dependencies:")
        print("   pip install sentence-transformers")
        print("4. Verify files exist and are readable")

if __name__ == "__main__":
    main()