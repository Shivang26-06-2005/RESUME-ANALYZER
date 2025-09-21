# Resume-JD Semantic Matching System
# Architecture: Gemma3 1B (Text Generation) + Qwen3-Embedding-4B (Embeddings) + Chroma Vector Store
# Tech Stack: Python, LangChain, LangGraph, LangSmith, Ollama, HuggingFace, spaCy
# 100% FREE - No API costs!

import os
import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import re
from datetime import datetime
import time

# Core libraries
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymupdf  # fitz
import docx2txt
from fuzzywuzzy import fuzz

# LangChain imports - Updated to latest versions
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# LangGraph imports
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# LangSmith for observability
from langsmith import Client

# Pydantic models for structured output
from pydantic import BaseModel, Field

# Sentence Transformers for embeddings (fallback)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("sentence-transformers not available. Install with: pip install sentence-transformers")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
    raise

@dataclass
class MatchingResult:
    """Result structure for resume-JD matching"""
    relevance_score: float
    missing_skills: List[str]
    missing_projects: List[str]
    missing_certifications: List[str]
    verdict: str
    suggestions: List[str]
    semantic_similarity: float
    keyword_similarity: float

class ParsedResume(BaseModel):
    """Structured resume data"""
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
    """Structured job description data"""
    role_title: str = ""
    must_have_skills: List[str] = []
    good_to_have_skills: List[str] = []
    qualifications: List[str] = []
    experience_required: str = ""
    raw_text: str = ""

class MatchingState(TypedDict):
    """State for LangGraph workflow"""
    resume: ParsedResume
    jd: ParsedJD
    keyword_score: float
    semantic_score: float
    skill_match_score: float
    final_score: float
    missing_items: Dict[str, List[str]]
    suggestions: List[str]
    verdict: str
    stage: str

class DocumentParser:
    """Handles parsing of PDF and DOCX documents"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = pymupdf.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX using docx2txt"""
        try:
            text = docx2txt.process(docx_path)
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,;:()\-@]', '', text)
        return text.strip()

class ResumeParser:
    """Parse resume content using LangChain + Gemma3 1B"""
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=ParsedResume)
        
        # Create LangChain prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert resume parser that ONLY outputs valid JSON. Extract structured information from the resume text.
            IMPORTANT: Format each experience, education, and project as a SINGLE STRING, not as an object.
            
            Focus on identifying:
            - Personal information (name, email, phone)
            - Technical skills (as list of strings)
            - Work experience (each job as a single descriptive string)
            - Education (each degree as a single descriptive string)
            - Projects (each project as a single descriptive string)
            - Certifications (as list of strings)
            
            Format your response EXACTLY as follows:
            {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "123-456-7890",
                "skills": ["Python", "Java", "AWS"],
                "experience": [
                    "Senior Developer at XYZ Corp (2020-Present): Led development of cloud infrastructure",
                    "Software Engineer at ABC Inc (2018-2020): Developed microservices architecture"
                ],
                "education": [
                    "MS Computer Science, Stanford University (2015-2017)",
                    "BS Computer Engineering, MIT (2011-2015)"
                ],
                "projects": [
                    "Cloud Migration Project: Led team of 5 to migrate legacy system to AWS",
                    "AI Chatbot: Developed customer service bot using Python and NLP"
                ],
                "certifications": ["AWS Solutions Architect", "PMP"]
            }

            CRITICAL RULES:
            1. ALL array items must be strings, not objects
            2. Format experience as "Title at Company (Date): Description"
            3. Format education as "Degree, Institution (Date)"
            4. Format projects as "Name: Description"
            5. No nested objects allowed
            
            {format_instructions}"""),
            HumanMessage(content="Resume text: {resume_text}")
        ])
        
        # Create chain using new pattern
        self.chain = self.prompt | self.llm
    
    def parse_resume(self, resume_text: str) -> ParsedResume:
        """Parse resume text into structured format using LangChain"""
        try:
            # Use spaCy for basic entity extraction
            doc = nlp(resume_text)
            
            # Extract entities using spaCy
            emails = []
            phones = []
            persons = []
            
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text.split()) <= 3:
                    persons.append(ent.text)
            
            # Find emails and phones using regex (fixed regex flags)
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            phone_pattern = r'\b[\+]?[1-9]?[0-9]{2,3}[-.\s]?[(]?[0-9]{1,4}[)]?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}\b'
            
            emails = re.findall(email_pattern, resume_text, re.IGNORECASE)
            phones = re.findall(phone_pattern, resume_text)
            
            # Use LangChain to process with Gemma3 1B
            response = self.chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "resume_text": resume_text[:3000]  # Limit for token constraints
            })
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = response[json_start:json_end]
                    parsed_data = json.loads(json_str)
                    
                    parsed_resume = ParsedResume(
                        name=parsed_data.get("name", persons[0] if persons else ""),
                        email=parsed_data.get("email", emails[0] if emails else ""),
                        phone=parsed_data.get("phone", phones[0] if phones else ""),
                        skills=parsed_data.get("skills", []),
                        experience=parsed_data.get("experience", []),
                        education=parsed_data.get("education", []),
                        projects=parsed_data.get("projects", []),
                        certifications=parsed_data.get("certifications", []),
                        raw_text=resume_text
                    )
                    
                    return parsed_resume
                else:
                    raise ValueError("Could not extract JSON from response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Could not parse JSON response: {e}")
                # Fallback to basic parsing
                return self._fallback_parse_resume(resume_text, emails, phones, persons)
            
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            return self._fallback_parse_resume(resume_text, emails, phones, persons)
    
    def _fallback_parse_resume(self, text: str, emails: List[str], phones: List[str], persons: List[str]) -> ParsedResume:
        """Fallback parsing using enhanced regex patterns"""
        # Comprehensive skill extraction
        skills = []
        
        # Technical Skills
        tech_pattern = r'\b(?:python|java|javascript|typescript|react|angular|vue|node\.js|express|django|flask|sql|mysql|postgresql|mongodb|docker|kubernetes|aws|azure|gcp|git|html|css|sass|webpack|jenkins|terraform|ansible|php|ruby|scala|rust|swift|kotlin|matlab|r|tensorflow|pytorch|scikit-learn|pandas|numpy|spring|hibernate|junit|selenium|cypress|redux|graphql|rest|soap|microservices|agile|scrum)\b'
        
        # Teaching/Education Skills
        education_pattern = r'\b(?:teaching|instruction|curriculum|lesson plan|classroom management|student assessment|differentiated instruction|special education|iep|education technology|smart board|blackboard|canvas|online teaching|distance learning|stem|student engagement|classroom discipline|early childhood|elementary education|secondary education|substitute teaching|tutoring|mentoring|pedagogy|educational psychology|student development|literacy|mathematics instruction|science education|social studies|special needs|individualized learning|group instruction|student evaluation)\b'
        
        # General Professional Skills
        prof_pattern = r'\b(?:leadership|management|communication|presentation|problem solving|analytical|research|teamwork|project management|time management|organization|planning|coordination|training|mentoring|coaching|reporting|documentation|analysis|strategy|customer service|client relations|collaboration|supervision|budgeting|scheduling|quality assurance|process improvement|resource management)\b'
        
        # Find all skills using the patterns
        for pattern in [tech_pattern, education_pattern, prof_pattern]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend([match.lower() for match in matches])
        
        # Extract skills mentioned with common skill indicators
        skill_indicators = r'(?:proficient in|experienced with|skilled in|knowledge of|expertise in|competent in|trained in|certified in|specializing in|background in)\s+([\w\s,]+?)(?:\.|\n|;|$)'
        skill_matches = re.finditer(skill_indicators, text, re.IGNORECASE)
        for match in skill_matches:
            if match.group(1):
                skill_list = match.group(1).strip().split(',')
                skills.extend([skill.strip().lower() for skill in skill_list if skill.strip()])
        
        return ParsedResume(
            name=persons[0] if persons else "",
            email=emails[0] if emails else "",
            phone=phones[0] if phones else "",
            skills=list(set(skills)),  # Remove duplicates
            raw_text=text
        )

class JDParser:
    """Parse job description using LangChain + Gemma3 1B"""
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=ParsedJD)
        
        # Create LangChain prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert job description parser. Extract structured information from the JD text.
            Focus on identifying:
            - Role/Job title
            - Must-have skills (required/mandatory skills)
            - Good-to-have skills (preferred/nice-to-have skills)
            - Educational qualifications
            - Experience requirements
            
            Return the information in the following JSON format:
            {{
                "role_title": "job title",
                "must_have_skills": ["skill1", "skill2", "skill3"],
                "good_to_have_skills": ["skill1", "skill2"],
                "qualifications": ["qualification1", "qualification2"],
                "experience_required": "experience description"
            }}
            
            {format_instructions}"""),
            HumanMessage(content="Job Description: {jd_text}")
        ])
        
        # Create chain using new pattern
        self.chain = self.prompt | self.llm
    
    def parse_jd(self, jd_text: str) -> ParsedJD:
        """Parse job description into structured format using LangChain"""
        try:
            # Use LangChain to process with Gemma3 1B
            response = self.chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "jd_text": jd_text[:3000]  # Limit for token constraints
            })
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = response[json_start:json_end]
                    parsed_data = json.loads(json_str)
                    
                    parsed_jd = ParsedJD(
                        role_title=parsed_data.get("role_title", ""),
                        must_have_skills=parsed_data.get("must_have_skills", []),
                        good_to_have_skills=parsed_data.get("good_to_have_skills", []),
                        qualifications=parsed_data.get("qualifications", []),
                        experience_required=parsed_data.get("experience_required", ""),
                        raw_text=jd_text
                    )
                    
                    return parsed_jd
                else:
                    raise ValueError("Could not extract JSON from response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Could not parse JSON response: {e}")
                return ParsedJD(raw_text=jd_text)
            
        except Exception as e:
            logger.error(f"Error parsing JD: {e}")
            return ParsedJD(raw_text=jd_text)

class KeywordMatcher:
    """Handle keyword and hard matching between resume and JD"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            stop_words='english',
            max_features=1000
        )
    
    def calculate_keyword_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculate TF-IDF based keyword similarity (BM25 alternative)"""
        try:
            documents = [resume_text, jd_text]
            tfidf_matrix = self.tfidf.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(similarity[0][0])
        except Exception as e:
            logger.error(f"Error calculating keyword similarity: {e}")
            return 0.0
    
    def fuzzy_match_skills(self, resume_skills: List[str], required_skills: List[str]) -> Tuple[List[str], float]:
        """Perform fuzzy matching of skills with semantic grouping"""
        if not required_skills or not resume_skills:
            return [], 0.0
            
        matched_skills = []
        total_score = 0
        
        # Define skill synonyms and related terms
        skill_groups = {
            'teaching': ['teach', 'instruction', 'education', 'classroom', 'tutoring', 'training'],
            'development': ['develop', 'programming', 'coding', 'software', 'engineering'],
            'analysis': ['analyze', 'research', 'data', 'assessment', 'evaluation'],
            'management': ['manage', 'coordinate', 'supervise', 'lead', 'direct'],
            'communication': ['communicate', 'present', 'write', 'speak', 'facilitate']
        }
        
        # Convert all skills to lowercase for matching
        resume_skills_lower = [skill.lower().strip() for skill in resume_skills]
        required_skills_lower = [skill.lower().strip() for skill in required_skills]
        
        for req_skill in required_skills_lower:
            best_match_score = 0
            matched_skill = None
            
            # Check for exact match first
            if req_skill in resume_skills_lower:
                best_match_score = 100
                matched_skill = req_skill
            else:
                # Check for skill group matches
                req_skill_words = req_skill.split()
                for resume_skill in resume_skills_lower:
                    # Try semantic group matching
                    semantic_match = False
                    for group, related_terms in skill_groups.items():
                        if (any(term in req_skill for term in related_terms) and 
                            any(term in resume_skill for term in related_terms)):
                            semantic_match = True
                            break
                    
                    # Calculate fuzzy match score
                    score = fuzz.token_sort_ratio(req_skill, resume_skill)
                    if semantic_match:
                        score = max(score, 85)  # Boost score for semantic matches
                        
                    if score > best_match_score:
                        best_match_score = score
                        matched_skill = resume_skill
            
            # Consider partial word matches for compound skills
            if best_match_score < 85:
                req_words = set(req_skill.split())
                for resume_skill in resume_skills_lower:
                    res_words = set(resume_skill.split())
                    word_match_ratio = len(req_words.intersection(res_words)) / len(req_words)
                    if word_match_ratio > 0.5:  # If more than half the words match
                        best_match_score = max(best_match_score, 80)
                        matched_skill = resume_skill
            
            if best_match_score >= 80:  # Slightly relaxed threshold with better matching
                matched_skills.append(req_skill)
                total_score += best_match_score
        
        # Calculate final score with emphasis on matching critical skills
        if required_skills:
            match_ratio = len(matched_skills) / len(required_skills)
            quality_score = total_score / (len(required_skills) * 100) if matched_skills else 0
            
            # Weighted scoring that emphasizes matching most important skills
            final_score = (match_ratio * 0.7 + quality_score * 0.3)
        else:
            final_score = 0.0
        
        return matched_skills, round(final_score, 3)

class SemanticMatcher:
    """Handle semantic matching using embeddings and LLM"""
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        self.embeddings = None
        self._initialize_embeddings()
        
        # Create feedback chain
        feedback_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert technical recruiter and career advisor with deep expertise in analyzing resume-job fit for technical roles. Your task is to provide highly specialized, technically precise feedback that demonstrates deep understanding of both the role requirements and industry standards.

            Analyze and provide feedback in the following JSON format, ensuring extreme specificity in every response:
            {
                "missing_skills": [
                    // Generate highly specific technical feedback about missing skills:
                    // Example: "While you have basic Python experience, this ML Engineering role specifically requires expertise in PyTorch's distributed training capabilities and CUDA optimization, which are not evident in your background"
                    // Example: "Your Kubernetes experience appears limited to basic pod management, but this Senior DevOps role requires advanced skills in custom controller development and operator patterns using the Kubernetes Go client"
                ],
                "missing_projects": [
                    // Provide technically detailed project suggestions that precisely match role requirements:
                    // Example: "Developing a microservices-based application using gRPC for inter-service communication and implementing circuit breakers with Istio would demonstrate the architectural expertise required for this Platform Engineer role"
                    // Example: "Creating an end-to-end MLOps pipeline using DVC for versioning, MLflow for experiment tracking, and Kubeflow for orchestration would showcase the practical MLOps skills needed for this position"
                ],
                "missing_certifications": [
                    // Recommend specific certification paths with clear reasoning:
                    // Example: "The Google Professional Machine Learning Engineer certification would validate your ability to design scalable ML architectures on GCP, a critical requirement for this ML Infrastructure role"
                    // Example: "Obtaining the Certified Kubernetes Security Specialist (CKS) certification would demonstrate your expertise in container security hardening and runtime security enforcement, which is essential for this Cloud Security Engineer position"
                ],
                "suggestions": [
                    // Provide highly specific, technically detailed improvement suggestions:
                    // Example: "Gain hands-on experience with Terraform's advanced features like custom providers and remote state management using S3 backend with state locking via DynamoDB, which are key requirements for this Infrastructure Engineer role"
                    // Example: "Develop expertise in advanced React patterns such as render props, higher-order components, and custom hooks, particularly focusing on performance optimization using useMemo and useCallback for complex component hierarchies"
                ]
            }

            Analysis Requirements:
            1. Technical Depth: Provide extremely detailed technical feedback that demonstrates expert knowledge of tools, frameworks, and industry best practices
            2. Role Specificity: Tailor every piece of feedback to the exact technical requirements and seniority level of the role
            3. Skill Gap Analysis: Analyze not just missing skills, but also the depth of existing skills versus required proficiency levels
            4. Industry Context: Include feedback about relevant industry trends and emerging technologies specific to the role
            5. Implementation Detail: When suggesting projects or improvements, include specific technical implementation details
            6. Architectural Understanding: For senior roles, emphasize system design and architectural expertise requirements
            7. Modern Practices: Reference current best practices, modern tooling, and contemporary development approaches
            8. Quantification: Where possible, include specific metrics or benchmarks for required expertise levels

            Response Requirements:
            1. NEVER use generic phrases like "good understanding" or "experience with" - always be specific about the exact technical capabilities required
            2. Include specific versions, tools, and frameworks when discussing technologies
            3. Reference concrete technical concepts, patterns, and methodologies
            4. Explain the technical reasoning behind each recommendation
            5. If a category has no missing items, use an empty array []
            6. Ensure each suggestion is immediately actionable and technically precise
            """),
            HumanMessage(content="""
            Job Requirements:
            Role: {role_title}
            Must-have skills: {must_have_skills}
            Good-to-have skills: {good_to_have_skills}
            Qualifications: {qualifications}
            
            Resume Summary:
            Skills: {resume_skills}
            Projects: {resume_projects}
            Certifications: {resume_certifications}
            Experience: {resume_experience}
            
            Provide specific feedback for improvement:
            """)
        ])
        
        self.feedback_chain = feedback_prompt | self.llm
    
    def _initialize_embeddings(self):
        """Initialize Qwen3-Embedding-4B model with proper error handling and retry logic"""
        # Try the specified model first, then fallbacks
        model_variants = [
            "dengcao/Qwen3-Embedding-4B:Q4_K_M",  # Primary choice as specified
            #"nomic-embed-text",  # Alternative 1
            "mxbai-embed-large"  # Alternative 2
        ]
        
        for model_name in model_variants:
            try:
                logger.info(f"Attempting to initialize embedding model: {model_name}")
                
                # Initialize embeddings with specific model
                embeddings = OllamaEmbeddings(
                    model=model_name,
                    base_url="http://localhost:11434"
                )
                
                # Test the embedding model with a simple query
                logger.info(f"Testing embedding model {model_name}...")
                test_text = "test embedding functionality"
                test_embedding = embeddings.embed_query(test_text)
                
                if test_embedding and len(test_embedding) > 0:
                    logger.info(f"Successfully initialized {model_name} for embeddings (dimension: {len(test_embedding)})")
                    self.embeddings = embeddings
                    return
                else:
                    logger.warning(f"Model {model_name} returned empty embeddings")
                    
            except Exception as e:
                logger.warning(f"Could not initialize {model_name}: {e}")
                continue
        
        # If all models fail, try sentence-transformers as fallback
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Falling back to sentence-transformers...")
                # Use lightweight sentence transformer model
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.sentence_model = sentence_model
                logger.info("Initialized sentence-transformers fallback")
                return
            except Exception as e:
                logger.error(f"Sentence-transformers fallback failed: {e}")
        
        # Final fallback - no embeddings
        logger.error("No embedding models available. Semantic similarity will return 0.0")
        logger.error("Please ensure you have installed one of these models:")
        logger.error("1. ollama pull qwen:7b")
        logger.error("2. ollama pull nomic-embed-text")  
        logger.error("3. ollama pull mxbai-embed-large")
        logger.error("4. pip install sentence-transformers")
    
    def calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculate semantic similarity using available embeddings and normalize to 0-1 scale"""
        # Truncate texts to reasonable length
        resume_text = resume_text[:1500]
        jd_text = jd_text[:1500]
        
        try:
            # Try Ollama embeddings first
            if self.embeddings:
                logger.debug("Using Ollama embeddings for semantic similarity")
                resume_embedding = self.embeddings.embed_query(resume_text)
                jd_embedding = self.embeddings.embed_query(jd_text)
                
                if resume_embedding and jd_embedding and len(resume_embedding) > 0 and len(jd_embedding) > 0:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        np.array(resume_embedding).reshape(1, -1),
                        np.array(jd_embedding).reshape(1, -1)
                    )
                    
                    # Normalize the similarity score from [-1,1] to [0,1]
                    normalized_score = (float(similarity[0][0]) + 1) / 2
                    logger.info(f"Semantic similarity calculated: {normalized_score:.3f}")
                    return round(normalized_score, 3)
                    
            # Try sentence-transformers fallback
            elif hasattr(self, 'sentence_model'):
                logger.debug("Using sentence-transformers for semantic similarity")
                resume_embedding = self.sentence_model.encode([resume_text])
                jd_embedding = self.sentence_model.encode([jd_text])
                
                similarity = cosine_similarity(resume_embedding, jd_embedding)
                normalized_score = (float(similarity[0][0]) + 1) / 2
                logger.info(f"Semantic similarity (sentence-transformers): {normalized_score:.3f}")
                return round(normalized_score, 3)
                
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
        
        # Final fallback - use keyword similarity as approximation
        logger.warning("No embedding models available, using keyword similarity as fallback")
        return self._calculate_fallback_similarity(resume_text, jd_text)
    
    def _calculate_fallback_similarity(self, resume_text: str, jd_text: str) -> float:
        """Fallback similarity calculation using TF-IDF"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            tfidf = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                stop_words='english',
                max_features=500
            )
            
            documents = [resume_text, jd_text]
            tfidf_matrix = tfidf.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            # Normalize to 0-1 range
            normalized_score = max(0.0, min(1.0, float(similarity[0][0])))
            logger.info(f"Fallback similarity calculated: {normalized_score:.3f}")
            return round(normalized_score, 3)
            
        except Exception as e:
            logger.error(f"Error in fallback similarity calculation: {e}")
            return 0.0
    
    def generate_feedback_and_suggestions(self, resume: ParsedResume, jd: ParsedJD, score: float) -> Tuple[Dict[str, List[str]], List[str]]:
        """Generate missing items and improvement suggestions using LangChain"""
        
        try:
            response = self.feedback_chain.invoke({
                "role_title": jd.role_title,
                "must_have_skills": ', '.join(jd.must_have_skills),
                "good_to_have_skills": ', '.join(jd.good_to_have_skills),
                "qualifications": ', '.join(jd.qualifications),
                "resume_skills": ', '.join(resume.skills),
                "resume_projects": ', '.join(resume.projects),
                "resume_certifications": ', '.join(resume.certifications),
                "resume_experience": ', '.join(resume.experience)
            })
            
            # Parse JSON response
            try:
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
                else:
                    raise ValueError("Could not extract JSON from response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Could not parse feedback JSON: {e}")
                return self._generate_basic_feedback(resume, jd)
            
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            return self._generate_basic_feedback(resume, jd)
    
    def _generate_basic_feedback(self, resume: ParsedResume, jd: ParsedJD) -> Tuple[Dict[str, List[str]], List[str]]:
        """Generate comprehensive feedback with detailed project and certification analysis"""
        # Initialize feedback collections
        missing_skills = []
        missing_projects = []
        missing_certs = []
        suggestions = []

        # 1. Skills Analysis
        for skill in jd.must_have_skills:
            if not any(fuzz.ratio(skill.lower(), res_skill.lower()) > 70 for res_skill in resume.skills):
                missing_skills.append(f"The role requires proficiency in {skill}, which is not demonstrated in the resume")

        # 2. Project Analysis
        required_keywords = set(k.lower() for k in jd.must_have_skills + jd.good_to_have_skills)
        
        if not resume.projects:
            missing_projects.append(f"The resume would benefit from adding projects that demonstrate practical experience in {jd.role_title} responsibilities")
        else:
            # Analyze each project for skill coverage
            demonstrated_skills = set()
            for project in resume.projects:
                project_words = set(word.lower() for word in project.split())
                demonstrated_skills.update(required_keywords.intersection(project_words))
            
            # Identify missing skill demonstrations
            missing_skill_demos = required_keywords - demonstrated_skills
            if missing_skill_demos:
                key_skills = list(missing_skill_demos)[:3]
                if key_skills:
                    missing_projects.append(
                        f"Consider adding projects that demonstrate practical experience with {', '.join(key_skills)} to align with job requirements"
                    )

        # 3. Certification Analysis
        role = jd.role_title.lower()
        cert_suggestions = []

        # Role-specific certification mapping
        cert_requirements = {
            "software": {
                "keywords": ["software", "developer", "engineer", "programming"],
                "certs": [
                    "Professional cloud certification (AWS, Azure, or GCP) to validate cloud computing expertise",
                    "Language-specific certifications relevant to the role's tech stack",
                    "Software architecture or design pattern certifications"
                ]
            },
            "data": {
                "keywords": ["data", "analytics", "scientist", "ml", "ai"],
                "certs": [
                    "Data Science certification to validate analytical capabilities",
                    "Big Data certifications (e.g., Hadoop, Spark) for large-scale data processing expertise",
                    "Machine Learning certifications to demonstrate AI capabilities"
                ]
            },
            "project": {
                "keywords": ["project", "manager", "lead", "management"],
                "certs": [
                    "Project Management Professional (PMP) certification to validate project management expertise",
                    "Agile certifications (Scrum Master, PRINCE2) to demonstrate methodology knowledge",
                    "Leadership and management certifications relevant to the industry"
                ]
            }
        }

        # Determine role-specific certifications
        for role_type, info in cert_requirements.items():
            if any(kw in role for kw in info["keywords"]):
                cert_suggestions.extend(info["certs"])
                break
        else:
            cert_suggestions.append(f"Industry-standard certifications relevant to {jd.role_title} roles")

        # Compare existing certifications
        if resume.certifications:
            existing_certs = set(cert.lower() for cert in resume.certifications)
            missing_certs = [
                cert for cert in cert_suggestions 
                if not any(exists in cert.lower() for exists in existing_certs)
            ]
        else:
            missing_certs = cert_suggestions

        # 4. Generate Final Output
        missing_items = {
            "skills": missing_skills[:5],
            "projects": missing_projects,
            "certifications": missing_certs
        }

        # 5. Generate Comprehensive Suggestions
        if missing_skills:
            suggestions.append(f"Enhance your expertise in {', '.join(s.split('in ')[1].split(',')[0] for s in missing_skills[:3])} through focused training and practical application")

        if missing_projects:
            suggestions.append("Develop portfolio projects that specifically demonstrate your capabilities in the required technical areas")

        if missing_certs:
            suggestions.append(f"Pursue relevant certifications to validate your expertise: {', '.join(c.split('(')[0].strip() for c in missing_certs[:2])}")

        # Add contextual improvement suggestions
        if resume.experience:
            suggestions.append("Strengthen your experience section by quantifying achievements and highlighting specific technical contributions")
        if resume.projects:
            suggestions.append("Enhance project descriptions by including technical challenges overcome and measurable outcomes achieved")
        suggestions.append(f"Align your resume more closely with {jd.role_title} requirements by highlighting relevant technical accomplishments")

        return missing_items, suggestions

class ResumeJDMatcher:
    """Main orchestrator for resume-JD matching using LangGraph"""
    
    def __init__(self, model_name: str = "gemma3:1b", langsmith_api_key: str = None):
        # Initialize Ollama LLM through LangChain (updated import)
        self.llm = OllamaLLM(model=model_name, temperature=0.1)
        
        # Setup LangSmith if API key provided
        if langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
            self.langsmith_client = Client()
        else:
            self.langsmith_client = None
        
        # Initialize components with LangChain (order matters!)
        self.document_parser = DocumentParser()
        self.resume_parser = ResumeParser(self.llm)
        self.jd_parser = JDParser(self.llm)
        self.keyword_matcher = KeywordMatcher()
        self.semantic_matcher = SemanticMatcher(self.llm)  # Initialize before vectorstore
        
        # Only semantic similarity is used for scoring
        self.weights = {
            "semantic_weight": 1.0
        }
        
        # Build LangGraph workflow (semantic only)
        self.workflow = self._build_langgraph_workflow()

        # Setup Chroma vector store with persistence (after semantic_matcher)
        self.vectorstore = self._setup_chroma_vectorstore()
    
    def _setup_chroma_vectorstore(self) -> Chroma:
        """Setup Chroma vector store with persistence"""
        try:
            # Create persist directory if it doesn't exist
            persist_directory = "./chroma_db"
            os.makedirs(persist_directory, exist_ok=True)
            
            # Use the same embedding method as SemanticMatcher
            if self.semantic_matcher.embeddings:
                embedding_function = self.semantic_matcher.embeddings
                logger.info("Using Ollama embeddings for Chroma vectorstore")
            elif hasattr(self.semantic_matcher, 'sentence_model'):
                # Create a wrapper for sentence-transformers
                from langchain_community.embeddings import SentenceTransformerEmbeddings
                embedding_function = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
                logger.info("Using sentence-transformers for Chroma vectorstore")
            else:
                # Use a simple embedding function for fallback
                embedding_function = None
                logger.warning("Using Chroma without embedding function")
            
            # Initialize Chroma with persistence
            if embedding_function:
                vectorstore = Chroma(
                    embedding_function=embedding_function,
                    persist_directory=persist_directory,
                    collection_name="resume_jd_matches"
                )
            else:
                # Create a basic vectorstore without embeddings
                vectorstore = None
                logger.warning("Chroma vectorstore disabled due to embedding issues")
            
            logger.info("Chroma vectorstore initialized successfully")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error setting up Chroma vectorstore: {e}")
            # Return None if vectorstore fails
            return None
    
    def _build_langgraph_workflow(self) -> StateGraph:
        """Build LangGraph workflow for resume-JD matching (semantic only)"""
        workflow = StateGraph(MatchingState)
        workflow.add_node("semantic_matching", self._semantic_matching_node)
        workflow.add_node("score_calculation", self._score_calculation_node)
        workflow.add_node("feedback_generation", self._feedback_generation_node)
        workflow.add_node("verdict_generation", self._verdict_generation_node)
        workflow.add_edge("semantic_matching", "score_calculation")
        workflow.add_edge("score_calculation", "feedback_generation")
        workflow.add_edge("feedback_generation", "verdict_generation")
        workflow.add_edge("verdict_generation", END)
        workflow.set_entry_point("semantic_matching")
        return workflow.compile()
    
    def _semantic_matching_node(self, state: MatchingState) -> MatchingState:
        """Perform semantic matching using embeddings"""
        semantic_score = self.semantic_matcher.calculate_semantic_similarity(
            state["resume"].raw_text, state["jd"].raw_text
        )
        state["semantic_score"] = semantic_score
        state["stage"] = "semantic_matching_complete"
        logger.info(f"Semantic matching score: {semantic_score:.3f}")
        return state
    
    def _score_calculation_node(self, state: MatchingState) -> MatchingState:
        """Final score is just the semantic similarity (embeddings)"""
        final_score = state["semantic_score"]
        state["final_score"] = min(round(final_score * 100, 2), 100)
        state["stage"] = "score_calculation_complete"
        logger.info(f"Final score: {state['final_score']:.2f}")
        return state
    
    def _feedback_generation_node(self, state: MatchingState) -> MatchingState:
        """Generate feedback and missing items"""
        missing_items, suggestions = self.semantic_matcher.generate_feedback_and_suggestions(
            state["resume"], state["jd"], state["final_score"]
        )
        
        state["missing_items"] = missing_items
        state["suggestions"] = suggestions
        state["stage"] = "feedback_generation_complete"
        return state
    
    def _verdict_generation_node(self, state: MatchingState) -> MatchingState:
        """Generate final verdict with stricter thresholds"""
        score = state["final_score"]
        
        if score >= 85:
            verdict = "High"
        elif score >= 70:
            verdict = "Medium"
        else:
            verdict = "Low"
        
        state["verdict"] = verdict
        state["stage"] = "verdict_generation_complete"
        logger.info(f"Final verdict: {verdict}")
        return state
    
    def match_resume_to_jd(self, resume_path: str, jd_path: str) -> MatchingResult:
        """Main method to match resume to job description using LangGraph workflow"""
        
        # Extract text from resume file
        if resume_path.lower().endswith('.pdf'):
            resume_text = self.document_parser.extract_text_from_pdf(resume_path)
        elif resume_path.lower().endswith(('.docx', '.doc')):
            resume_text = self.document_parser.extract_text_from_docx(resume_path)
        else:
            raise ValueError("Unsupported resume file format. Please use PDF or DOCX.")
        
        if not resume_text:
            raise ValueError("Could not extract text from resume file.")
        
        # Extract text from JD file
        if jd_path.lower().endswith('.pdf'):
            jd_text = self.document_parser.extract_text_from_pdf(jd_path)
        elif jd_path.lower().endswith(('.docx', '.doc')):
            jd_text = self.document_parser.extract_text_from_docx(jd_path)
        else:
            raise ValueError("Unsupported JD file format. Please use PDF or DOCX.")
        
        if not jd_text:
            raise ValueError("Could not extract text from job description file.")
        
        # Parse resume and JD using LangChain
        parsed_resume = self.resume_parser.parse_resume(resume_text)
        parsed_jd = self.jd_parser.parse_jd(jd_text)
        
        # Initialize state for LangGraph
        initial_state = MatchingState(
            resume=parsed_resume,
            jd=parsed_jd,
            keyword_score=0.0,
            semantic_score=0.0,
            skill_match_score=0.0,
            final_score=0.0,
            missing_items={},
            suggestions=[],
            verdict="",
            stage="initialized"
        )
        
        # Run LangGraph workflow
        logger.info("Starting LangGraph workflow...")
        final_state = self.workflow.invoke(initial_state)
        
        # Store results in Chroma vector database for future retrieval
        self._store_in_vectorstore(parsed_resume, parsed_jd, final_state)
        
        # Create result
        result = MatchingResult(
            relevance_score=final_state['final_score'],
            missing_skills=final_state['missing_items'].get("skills", []),
            missing_projects=final_state['missing_items'].get("projects", []),
            missing_certifications=final_state['missing_items'].get("certifications", []),
            verdict=final_state['verdict'],
            suggestions=final_state['suggestions'],
            semantic_similarity=final_state['semantic_score'],
            keyword_similarity=final_state['keyword_score']
        )
        
        return result
    
    def _store_in_vectorstore(self, resume: ParsedResume, jd: ParsedJD, final_state: MatchingState):
        """Store matching results in Chroma vector store"""
        try:
            if not self.vectorstore:
                return
                
            # Create document text for storage
            doc_text = f"Resume: {resume.name} | Skills: {', '.join(resume.skills)} | Job: {jd.role_title}"
            
            # Metadata for the document
            metadata = {
                "resume_name": resume.name or "Unknown",
                "resume_email": resume.email,
                "job_title": jd.role_title,
                "score": final_state['final_score'],
                "verdict": final_state['verdict'],
                "timestamp": datetime.now().isoformat(),
                "skills_count": len(resume.skills),
                "missing_skills": ', '.join(final_state['missing_items'].get("skills", []))
            }
            
            # Add to Chroma vectorstore
            self.vectorstore.add_texts(
                texts=[doc_text],
                metadatas=[metadata],
                ids=[f"match_{resume.name}_{jd.role_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"]
            )
            
            # Persist the vectorstore
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()
            
            logger.info(f"Stored matching result in Chroma vectorstore for {resume.name}")
            
        except Exception as e:
            logger.warning(f"Could not store in Chroma vectorstore: {e}")

# Batch processing for multiple resumes
class BatchProcessor:
    """Handle batch processing of multiple resumes against single JD"""
    
    def __init__(self, matcher: ResumeJDMatcher):
        self.matcher = matcher
    
    def process_multiple_resumes(self, resume_folder: str, jd_path: str) -> List[Dict[str, Any]]:
        """Process multiple resumes against a single JD"""
        results = []
        resume_folder_path = Path(resume_folder)
        
        # Get all resume files
        resume_files = []
        for ext in ['*.pdf', '*.docx', '*.doc']:
            resume_files.extend(resume_folder_path.glob(ext))
        
        for resume_file in resume_files:
            try:
                result = self.matcher.match_resume_to_jd(str(resume_file), jd_path)
                results.append({
                    'resume_file': resume_file.name,
                    'resume_path': str(resume_file),
                    'result': result
                })
                logger.info(f"Processed: {resume_file.name} - Score: {result.relevance_score:.2f}")
            except Exception as e:
                logger.error(f"Error processing {resume_file.name}: {e}")
                results.append({
                    'resume_file': resume_file.name,
                    'resume_path': str(resume_file),
                    'error': str(e)
                })
        
        return results
    
    def filter_and_rank_resumes(self, results: List[Dict[str, Any]], 
                               min_score: float = 50.0) -> List[Dict[str, Any]]:
        """Filter and rank resumes by score"""
        # Filter out errors and low scores
        valid_results = [r for r in results if 'result' in r and r['result'].relevance_score >= min_score]
        
        # Sort by relevance score (descending)
        valid_results.sort(key=lambda x: x['result'].relevance_score, reverse=True)
        
        return valid_results
    
    def search_resumes_by_criteria(self, job_role: str = "", 
                                  min_score: float = 50.0, 
                                  skills: List[str] = None) -> List[Dict]:
        """Search and filter resumes using Chroma vector store"""
        try:
            if self.matcher.vectorstore:
                # Build search query
                search_terms = []
                if job_role:
                    search_terms.append(f"job: {job_role}")
                if skills:
                    search_terms.append(f"skills: {', '.join(skills)}")
                
                search_query = " ".join(search_terms) if search_terms else "resume"
                
                # Search similar resumes using semantic search
                docs = self.matcher.vectorstore.similarity_search(
                    search_query, 
                    k=20,
                    filter={"score": {"$gte": min_score}} if min_score > 0 else None
                )
                
                results = []
                for doc in docs:
                    metadata = doc.metadata
                    if metadata.get('score', 0) >= min_score:
                        results.append({
                            'resume_name': metadata.get('resume_name', 'Unknown'),
                            'job_title': metadata.get('job_title', ''),
                            'score': metadata.get('score', 0),
                            'verdict': metadata.get('verdict', ''),
                            'skills_count': metadata.get('skills_count', 0),
                            'missing_skills': metadata.get('missing_skills', ''),
                            'timestamp': metadata.get('timestamp', ''),
                            'content': doc.page_content
                        })
                
                # Sort by score
                results.sort(key=lambda x: x['score'], reverse=True)
                return results
                
            else:
                logger.warning("Chroma vector store not available")
                return []
                
        except Exception as e:
            logger.error(f"Error searching resumes in Chroma: {e}")
            return []
    
    def get_analytics_from_vectorstore(self) -> Dict[str, Any]:
        """Get analytics and insights from stored resume matches"""
        try:
            if not self.matcher.vectorstore:
                return {"error": "Vectorstore not available"}
                
            # Get all documents from vectorstore
            all_docs = self.matcher.vectorstore.get()
            
            if not all_docs or not all_docs.get('metadatas'):
                return {"error": "No data found in vectorstore"}
            
            metadatas = all_docs['metadatas']
            
            # Calculate analytics
            total_resumes = len(metadatas)
            scores = [meta.get('score', 0) for meta in metadatas if 'score' in meta]
            
            analytics = {
                'total_resumes_processed': total_resumes,
                'average_score': sum(scores) / len(scores) if scores else 0,
                'highest_score': max(scores) if scores else 0,
                'lowest_score': min(scores) if scores else 0,
                'score_distribution': {
                    'high': len([s for s in scores if s >= 75]),
                    'medium': len([s for s in scores if 50 <= s < 75]),
                    'low': len([s for s in scores if s < 50])
                },
                'job_titles': {},
                'common_missing_skills': {},
                'top_candidates': []
            }
            
            # Analyze job titles
            for meta in metadatas:
                job_title = meta.get('job_title', 'Unknown')
                analytics['job_titles'][job_title] = analytics['job_titles'].get(job_title, 0) + 1
            
            # Analyze missing skills
            for meta in metadatas:
                missing_skills = meta.get('missing_skills', '')
                if missing_skills:
                    for skill in missing_skills.split(', '):
                        skill = skill.strip()
                        if skill:
                            analytics['common_missing_skills'][skill] = analytics['common_missing_skills'].get(skill, 0) + 1
            
            # Get top candidates
            scored_candidates = [(meta.get('resume_name', 'Unknown'), meta.get('score', 0), meta.get('job_title', '')) 
                               for meta in metadatas if 'score' in meta]
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            analytics['top_candidates'] = scored_candidates[:10]
            
            # Sort dictionaries
            analytics['job_titles'] = dict(sorted(analytics['job_titles'].items(), key=lambda x: x[1], reverse=True))
            analytics['common_missing_skills'] = dict(sorted(analytics['common_missing_skills'].items(), key=lambda x: x[1], reverse=True)[:10])
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting analytics from vectorstore: {e}")
            return {"error": str(e)}

# Additional utility functions for web application integration
def create_summary_report(results: List[Dict[str, Any]], output_file: str = "matching_report.json"):
    """Create a summary report of all matching results for placement team dashboard"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_resumes': len(results),
        'successful_matches': len([r for r in results if 'result' in r]),
        'failed_matches': len([r for r in results if 'error' in r]),
        'average_score': 0,
        'score_distribution': {'High': 0, 'Medium': 0, 'Low': 0},
        'top_candidates': [],
        'skill_gap_analysis': {},
        'detailed_results': []
    }
    
    successful_results = [r for r in results if 'result' in r]
    
    if successful_results:
        total_score = sum(r['result'].relevance_score for r in successful_results)
        report['average_score'] = total_score / len(successful_results)
        
        # Score distribution
        for result_data in successful_results:
            result = result_data['result']
            report['score_distribution'][result.verdict] += 1
        
        # Top candidates
        sorted_results = sorted(successful_results, key=lambda x: x['result'].relevance_score, reverse=True)
        report['top_candidates'] = [
            {
                'resume_file': r['resume_file'],
                'score': r['result'].relevance_score,
                'verdict': r['result'].verdict
            }
            for r in sorted_results[:10]
        ]
        
        # Skill gap analysis
        all_missing_skills = []
        for result_data in successful_results:
            all_missing_skills.extend(result_data['result'].missing_skills)
        
        skill_counts = {}
        for skill in all_missing_skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        report['skill_gap_analysis'] = dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Detailed results for dashboard
        for result_data in successful_results:
            result = result_data['result']
            report['detailed_results'].append({
                'resume_file': result_data['resume_file'],
                'score': result.relevance_score,
                'verdict': result.verdict,
                'missing_skills': result.missing_skills[:3],  # Top 3
                'suggestions': result.suggestions[:2],  # Top 2
                'semantic_similarity': result.semantic_similarity,
                'keyword_similarity': result.keyword_similarity
            })
    
    # Save report to file
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def export_shortlisted_candidates(results: List[Dict[str, Any]], 
                                min_score: float = 60.0, 
                                output_file: str = "shortlisted_candidates.json"):
    """Export shortlisted candidates for placement team"""
    
    shortlisted = [r for r in results if 'result' in r and r['result'].relevance_score >= min_score]
    shortlisted.sort(key=lambda x: x['result'].relevance_score, reverse=True)
    
    export_data = {
        'job_description': 'Job Description File',
        'shortlisting_criteria': f'Minimum score: {min_score}',
        'total_candidates': len(shortlisted),
        'generated_at': datetime.now().isoformat(),
        'candidates': []
    }
    
    for candidate in shortlisted:
        result = candidate['result']
        export_data['candidates'].append({
            'resume_file': candidate['resume_file'],
            'relevance_score': result.relevance_score,
            'verdict': result.verdict,
            'missing_skills': result.missing_skills,
            'suggestions_for_interview': result.suggestions[:3],
            'semantic_match': result.semantic_similarity,
            'keyword_match': result.keyword_similarity
        })
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return export_data

# Setup instructions for the system
def print_setup_instructions():
    """Print setup instructions for the system"""
    print("🚀 Setup Instructions for Resume-JD Matching System")
    print("Configuration: Gemma3 1B + Multiple Embedding Models")
    print("="*70)
    print()
    print("1. Install Ollama:")
    print("   - Download from: https://ollama.ai")
    print("   - Follow installation instructions for your OS")
    print()
    print("2. Install required Ollama models:")
    print("   ollama pull gemma3:1b                              # For text generation")
    print("   ollama pull dengcao/Qwen3-Embedding-4B:Q4_K_M     # Primary embedding model")
    print("   ollama pull nomic-embed-text                       # Alternative embedding")
    print("   ollama pull mxbai-embed-large                      # Alternative embedding")
    print("   (System will try these models in order)")
    print()
    print("3. Start Ollama server:")
    print("   ollama serve")
    print()
    print("4. Install Python dependencies:")
    print("   pip install langchain-ollama langchain-chroma langgraph")
    print("   pip install sentence-transformers  # Fallback embeddings")
    print("   pip install langchain langchain-community")
    print("   pip install spacy PyMuPDF docx2txt fuzzywuzzy")
    print("   pip install scikit-learn numpy pandas")
    print()
    print("5. Install spaCy English model:")
    print("   python -m spacy download en_core_web_sm")
    print()
    print("6. Optional - LangSmith for observability:")
    print("   - Sign up at: https://smith.langchain.com")
    print("   - Get API key and set in matcher initialization")
    print()
    print("✅ System Architecture:")
    print("- Text Generation: Gemma3 1B (via Ollama)")
    print("- Embeddings: Multiple fallback options")
    print("  1. Qwen 7B (primary)")
    print("  2. nomic-embed-text (alternative)")
    print("  3. mxbai-embed-large (alternative)")
    print("  4. sentence-transformers (fallback)")
    print("  5. TF-IDF similarity (final fallback)")
    print("- Vector Store: Chroma (persistent storage)")
    print("- Workflow: LangGraph")
    print()
    print("✅ Model Information:")
    print("- Gemma3 1B: ~1GB (text generation)")
    print("- Qwen 7B: ~4GB (embeddings)")
    print("- nomic-embed-text: ~274MB (alternative)")
    print("- mxbai-embed-large: ~669MB (alternative)")
    print()
    print("🔧 Troubleshooting:")
    print("- System automatically tries multiple embedding models")
    print("- Check 'ollama list' to verify models are installed")
    print("- Ensure Ollama service is running with 'ollama serve'")
    print("- If all embedding models fail, TF-IDF fallback is used")
    print()
    print("📊 Features:")
    print("- Robust embedding model fallback system")
    print("- Persistent vector storage with Chroma")
    print("- Batch processing capabilities")
    print("- Analytics and reporting")
    print("- 100% Free (no API costs)")
    print()
    print("🚀 Quick Test:")
    print("After setup, run: python your_script.py")
    print("The system will automatically test embedding models and use the best available.")

# Example usage and testing
def main():
    """Example usage of the Resume-JD Matching System with improved embedding handling"""
    
    # Initialize the matcher 
    # Optional: Add LangSmith API key for observability
    langsmith_api_key = None  # Replace with your LangSmith API key if you have one
    
    print("\n=== Initializing Resume-JD Matching System ===")
    print("Testing embedding models and initializing system...")
    
    try:
        matcher = ResumeJDMatcher(model_name="gemma3:1b", langsmith_api_key=langsmith_api_key)
        
        # Check which embedding model is being used
        if matcher.semantic_matcher.embeddings:
            print(f"✅ Using Ollama embeddings successfully")
        elif hasattr(matcher.semantic_matcher, 'sentence_model'):
            print(f"✅ Using sentence-transformers fallback")
        else:
            print(f"⚠️  Using TF-IDF similarity fallback")
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("\nPlease run with 'setup' argument for installation instructions:")
        print("python your_script.py setup")
        return
    
    # Example file paths (replace with actual file paths)
    resume_file_path = "resume - 3.pdf"  # Default resume file in workspace
    jd_file_path = "sample_jd_2.pdf"    # Default JD file in workspace
    
    try:
        print(f"\n=== Resume-Job Description Matching Analysis ===")
        print(f"Resume: {resume_file_path}")
        print(f"Job Description: {jd_file_path}")
        
        result = matcher.match_resume_to_jd(resume_file_path, jd_file_path)
        
        # Display results
        print(f"\n📊 Overall Results:")
        print(f"Relevance Score: {result.relevance_score:.2f}/100")
        print(f"Match Level: {result.verdict} suitability")
        
        print(f"\n🔍 Detailed Analysis:")
        print(f"Semantic Similarity: {result.semantic_similarity:.3f}")
        print(f"Keyword Similarity: {result.keyword_similarity:.3f}")
        
        print("\n❌ Gap Analysis - Missing Skills:")
        if result.missing_skills:
            for i, skill in enumerate(result.missing_skills[:5], 1):
                print(f"  {i}. {skill}")
        else:
            print("  No significant skill gaps identified")
        
        print("\n📁 Recommended Projects:")
        if result.missing_projects:
            for i, project in enumerate(result.missing_projects[:3], 1):
                print(f"  {i}. {project}")
        else:
            print("  Current projects align well with requirements")
        
        print("\n🏆 Recommended Certifications:")
        if result.missing_certifications:
            for i, cert in enumerate(result.missing_certifications[:3], 1):
                print(f"  {i}. {cert}")
        else:
            print("  Certification requirements are met")
        
        print("\n💡 Improvement Suggestions:")
        if result.suggestions:
            for i, suggestion in enumerate(result.suggestions[:5], 1):
                print(f"  {i}. {suggestion}")
        else:
            print("  Resume is well-aligned with job requirements")
            
        print(f"\n✅ Analysis complete! Results stored in Chroma vector database.")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please ensure the resume and job description files exist in the current directory")
    except Exception as e:
        print(f"❌ Error in matching process: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Check if Gemma3 1B is installed: ollama list")
        print("3. Verify file formats are supported (PDF/DOCX)")

# Batch processing example
def run_batch_example():
    """Example of batch processing multiple resumes"""
    print("\n=== Batch Processing Example ===")
    
    try:
        matcher = ResumeJDMatcher(model_name="gemma3:1b")
        batch_processor = BatchProcessor(matcher)
        
        # Process multiple resumes in a folder
        resume_folder = "./resumes"  # Folder containing resume files
        jd_path = "sample_jd.pdf"   # Single job description
        
        print(f"Processing resumes in folder: {resume_folder}")
        results = batch_processor.process_multiple_resumes(resume_folder, jd_path)
        
        # Filter and rank results
        shortlisted = batch_processor.filter_and_rank_resumes(results, min_score=60.0)
        
        print(f"\n📈 Batch Processing Results:")
        print(f"Total resumes processed: {len(results)}")
        print(f"Shortlisted candidates (>60%): {len(shortlisted)}")
        
        # Display top candidates
        print(f"\n🏆 Top Candidates:")
        for i, candidate in enumerate(shortlisted[:5], 1):
            result = candidate['result']
            print(f"  {i}. {candidate['resume_file']} - {result.relevance_score:.1f}% ({result.verdict})")
        
        # Generate reports
        report = create_summary_report(results, "batch_report.json")
        export_data = export_shortlisted_candidates(shortlisted, min_score=70.0, output_file="shortlisted.json")
        
        print(f"\n📄 Reports generated:")
        print(f"- Batch report: batch_report.json")
        print(f"- Shortlisted candidates: shortlisted.json")
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        print_setup_instructions()
    elif len(sys.argv) > 1 and sys.argv[1] == "batch":
        run_batch_example()
    else:
        main()