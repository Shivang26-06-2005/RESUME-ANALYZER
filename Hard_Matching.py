import spacy
import re
import pymupdf  # fitz
import docx2txt
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
from fuzzywuzzy import fuzz, process
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SkillMatch:
    skill: str
    match_type: str  # 'exact', 'fuzzy', 'contextual'
    confidence: float
    matched_text: str
    jd_context: str
    resume_context: str

@dataclass
class DocumentAnalysis:
    raw_text: str
    skills: Dict[str, Set[str]]
    experience_years: float
    education_level: str
    key_requirements: List[str]
    responsibilities: List[str]
    
@dataclass
class HardScoreResult:
    overall_score: float
    skill_matches: List[SkillMatch]
    missing_skills: List[str]
    technical_skills_score: float
    qualification_score: float
    experience_score: float
    keyword_density_score: float
    requirement_match_score: float
    breakdown: Dict[str, float]
    verdict: str  # 'High', 'Medium', 'Low'
    suggestions: List[str]
    jd_analysis: DocumentAnalysis
    resume_analysis: DocumentAnalysis

class DocumentParser:
    """Enhanced document parser for both PDF and DOCX files."""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF using PyMuPDF with better formatting."""
        try:
            doc = pymupdf.open(file_path)
            text = ""
            for page in doc:
                # Extract text with better formatting
                page_text = page.get_text()
                # Clean up extra whitespace but preserve line breaks
                page_text = re.sub(r'\n\s*\n', '\n\n', page_text)
                page_text = re.sub(r'[ \t]+', ' ', page_text)
                text += page_text + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX using docx2txt."""
        try:
            text = docx2txt.process(file_path)
            # Clean up the text
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """Extract text from PDF or DOCX file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return DocumentParser.extract_text_from_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return DocumentParser.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported: .pdf, .docx, .doc")

class JobDescriptionParser:
    """Specialized parser for job description structure."""
    
    def __init__(self):
        self.requirement_indicators = [
            r'(?:requirements?|qualifications?|skills?|must\s+have|required|essential)[:.]',
            r'(?:we\s+(?:are\s+)?looking\s+for|seeking|candidate\s+should|you\s+(?:should\s+)?have)',
            r'(?:mandatory|compulsory|obligatory|necessary|needed)'
        ]
        
        self.responsibility_indicators = [
            r'(?:responsibilities|duties|role|job\s+description|what\s+you\s+will\s+do)',
            r'(?:key\s+responsibilities|primary\s+duties|main\s+tasks)',
            r'(?:you\s+will|the\s+role\s+involves|day\s+to\s+day)'
        ]
    
    def extract_requirements(self, text: str) -> List[str]:
        """Extract requirements from job description."""
        requirements = []
        text_lower = text.lower()
        
        # Find requirement sections
        for pattern in self.requirement_indicators:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Extract text after the indicator
                start_pos = match.end()
                # Find the end of this section (next major heading or end of text)
                section_end = self._find_section_end(text, start_pos)
                section_text = text[start_pos:section_end]
                
                # Extract individual requirements
                req_items = self._parse_bullet_points(section_text)
                requirements.extend(req_items)
        
        # If no structured requirements found, extract from entire text
        if not requirements:
            requirements = self._extract_implicit_requirements(text)
        
        return list(set(requirements))  # Remove duplicates
    
    def extract_responsibilities(self, text: str) -> List[str]:
        """Extract responsibilities from job description."""
        responsibilities = []
        text_lower = text.lower()
        
        for pattern in self.responsibility_indicators:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start_pos = match.end()
                section_end = self._find_section_end(text, start_pos)
                section_text = text[start_pos:section_end]
                
                resp_items = self._parse_bullet_points(section_text)
                responsibilities.extend(resp_items)
        
        return list(set(responsibilities))
    
    def _find_section_end(self, text: str, start_pos: int, max_length: int = 1000) -> int:
        """Find the end of a section."""
        # Look for next major heading or bullet point section
        section_indicators = [
            r'\n\s*(?:responsibilities|requirements|qualifications|benefits|about)',
            r'\n\s*[A-Z][A-Za-z\s]{10,}:',  # Major headings
            r'\n\s*\d+\.',  # Numbered lists ending
        ]
        
        end_pos = start_pos + max_length
        for pattern in section_indicators:
            match = re.search(pattern, text[start_pos:], re.IGNORECASE)
            if match:
                end_pos = min(end_pos, start_pos + match.start())
        
        return min(end_pos, len(text))
    
    def _parse_bullet_points(self, text: str) -> List[str]:
        """Parse bullet points and list items from text."""
        items = []
        
        # Common bullet point patterns
        bullet_patterns = [
            r'(?:^|\n)\s*[-•*]\s*(.+?)(?=\n\s*[-•*]|\n\s*\n|\Z)',
            r'(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\n\s*\n|\Z)',
            r'(?:^|\n)\s*[a-zA-Z]\)\s*(.+?)(?=\n\s*[a-zA-Z]\)|\n\s*\n|\Z)',
        ]
        
        for pattern in bullet_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                item = match.group(1).strip()
                if len(item) > 10:  # Filter out very short items
                    items.append(item)
        
        # If no bullet points found, split by sentences
        if not items:
            sentences = re.split(r'[.!?]\s+', text)
            items = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return items[:10]  # Limit to 10 items
    
    def _extract_implicit_requirements(self, text: str) -> List[str]:
        """Extract requirements when no clear structure is present."""
        # Look for skill mentions and experience requirements
        requirements = []
        
        # Experience patterns
        exp_matches = re.findall(r'(\d+[\+\-]*\s*years?\s+(?:of\s+)?experience[^.]*)', text, re.IGNORECASE)
        requirements.extend(exp_matches)
        
        # Degree requirements
        degree_matches = re.findall(r'((?:bachelor|master|phd)[^.]*)', text, re.IGNORECASE)
        requirements.extend(degree_matches)
        
        # Technology requirements (sentences containing technology names)
        tech_patterns = [
            r'([^.]*(?:python|java|javascript|react|angular|sql|aws|docker)[^.]*)',
            r'([^.]*(?:experience|knowledge|proficient|skilled)\s+(?:in|with)[^.]*)'
        ]
        
        for pattern in tech_patterns:
            tech_matches = re.findall(pattern, text, re.IGNORECASE)
            requirements.extend(tech_matches)
        
        return requirements[:15]  # Limit to 15 items

class EnhancedSkillExtractor:
    """Enhanced skill extraction with better categorization."""
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Comprehensive skill patterns
        self.skill_patterns = {
            'programming_languages': [
                r'\b(?:python|java|javascript|typescript|c\+\+|c#|php|ruby|go|rust|kotlin|swift|scala|r|matlab|perl|bash|shell|powershell)\b',
                r'\b(?:js|ts|node\.?js|react\.?js|vue\.?js|angular\.?js|html5?|css3?)\b'
            ],
            'frameworks_libraries': [
                r'\b(?:django|flask|fastapi|spring|express|angular|react|vue|laravel|rails|nextjs|nuxt|gatsby)\b',
                r'\b(?:tensorflow|pytorch|keras|scikit-learn|pandas|numpy|opencv|nltk|spacy|matplotlib|seaborn)\b',
                r'\b(?:bootstrap|tailwind|jquery|axios|redux|vuex|mobx|jest|mocha|cypress)\b',
                r'\b(?:hibernate|jpa|spring\s+boot|asp\.net|\.net|entity\s+framework)\b'
            ],
            'databases': [
                r'\b(?:mysql|postgresql|mongodb|redis|elasticsearch|sqlite|oracle|sql\s+server|cassandra|dynamodb|neo4j)\b',
                r'\b(?:nosql|sql|database|db|rdbms|oltp|olap|data\s+warehouse|etl)\b'
            ],
            'cloud_devops': [
                r'\b(?:aws|azure|gcp|google\s+cloud|amazon\s+web\s+services|microsoft\s+azure|digital\s+ocean)\b',
                r'\b(?:docker|kubernetes|k8s|containerization|microservices|serverless|lambda)\b',
                r'\b(?:jenkins|gitlab\s+ci|github\s+actions|circleci|terraform|ansible|puppet|chef)\b',
                r'\b(?:ci/cd|devops|sre|site\s+reliability)\b'
            ],
            'tools_software': [
                r'\b(?:git|github|gitlab|bitbucket|svn|mercurial|version\s+control)\b',
                r'\b(?:jira|confluence|trello|asana|slack|teams|zoom|notion)\b',
                r'\b(?:vs\s+code|intellij|eclipse|pycharm|sublime|atom|vim|emacs)\b',
                r'\b(?:postman|insomnia|swagger|apache|nginx|linux|unix|windows|mac|ios|android)\b',
                r'\b(?:photoshop|illustrator|figma|sketch|adobe|office|excel|powerpoint)\b'
            ],
            'methodologies_concepts': [
                r'\b(?:agile|scrum|kanban|waterfall|devops|ci/cd|tdd|bdd|ddd)\b',
                r'\b(?:microservices|mvc|rest|restful|graphql|soap|api|sdk)\b',
                r'\b(?:machine\s+learning|deep\s+learning|ai|artificial\s+intelligence|data\s+science)\b',
                r'\b(?:blockchain|cryptocurrency|iot|big\s+data|analytics)\b'
            ],
            'soft_skills': [
                r'\b(?:leadership|communication|teamwork|problem\s+solving|analytical|creative)\b',
                r'\b(?:project\s+management|time\s+management|collaboration|mentoring)\b'
            ]
        }
        
        # Enhanced experience patterns
        self.experience_patterns = [
            r'(\d+(?:\.\d+)?)[\+\-\s]*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp|work)',
            r'(?:experience|exp|work)[:\s]*(\d+(?:\.\d+)?)[\+\-\s]*(?:years?|yrs?)',
            r'(\d+(?:\.\d+)?)[\+\-\s]*(?:years?|yrs?)\s*(?:in|with|of|as|working)',
            r'(?:over|more\s+than|above)\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
        ]
        
        # Enhanced education patterns
        self.education_patterns = [
            r'\b(?:bachelor|b\.?tech|b\.?e|b\.?sc|b\.?com|b\.?a|b\.?s|btech|be|bsc|bcom|ba|bs|undergraduate)\b',
            r'\b(?:master|m\.?tech|m\.?e|m\.?sc|m\.?com|m\.?a|m\.?s|mtech|me|msc|mcom|ma|ms|mba|graduate)\b',
            r'\b(?:phd|ph\.?d|doctorate|doctoral|postgraduate)\b',
            r'\b(?:diploma|certification|certificate|certified|associate|professional)\b'
        ]
    
    def analyze_document(self, text: str) -> DocumentAnalysis:
        """Comprehensive document analysis."""
        skills = self.extract_skills_from_text(text)
        experience_years = self.extract_experience_years(text)
        education_level = self.extract_education_level(text)
        
        # For job descriptions, extract requirements and responsibilities
        jd_parser = JobDescriptionParser()
        requirements = jd_parser.extract_requirements(text)
        responsibilities = jd_parser.extract_responsibilities(text)
        
        return DocumentAnalysis(
            raw_text=text,
            skills=skills,
            experience_years=experience_years,
            education_level=education_level,
            key_requirements=requirements,
            responsibilities=responsibilities
        )
    
    def extract_skills_from_text(self, text: str) -> Dict[str, Set[str]]:
        """Extract skills with improved accuracy."""
        text_lower = text.lower()
        extracted_skills = defaultdict(set)
        
        # Pattern-based extraction
        for category, patterns in self.skill_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    skill = match.group().strip()
                    if len(skill) > 1 and not skill.isdigit():
                        # Clean up the skill
                        skill = re.sub(r'[^\w\s+#.-]', '', skill)
                        if skill:
                            extracted_skills[category].add(skill)
        
        # NLP-based extraction
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract technical noun phrases
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower().strip()
                if self._is_technical_term(chunk_text):
                    extracted_skills['technical_terms'].add(chunk_text)
            
            # Extract technology-related named entities
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'GPE'] and self._is_technical_term(ent.text.lower()):
                    extracted_skills['entities'].add(ent.text.lower())
        
        # Remove empty sets and convert to regular dict
        return {k: v for k, v in extracted_skills.items() if v}
    
    def _is_technical_term(self, term: str) -> bool:
        """Enhanced technical term detection."""
        if len(term) < 2 or len(term) > 50:
            return False
        
        # Skip common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must'}
        if term.lower() in common_words:
            return False
        
        # Technical indicators
        technical_indicators = [
            'framework', 'library', 'language', 'database', 'server', 'cloud',
            'api', 'sdk', 'platform', 'tool', 'software', 'system', 'service',
            'technology', 'stack', 'environment', 'infrastructure'
        ]
        
        # Check for version numbers, file extensions, or technical patterns
        if re.search(r'\d+\.?\d*|js$|py$|sql$|db$|\.net$|\.com$', term):
            return True
        
        # Check for technical indicators
        if any(indicator in term for indicator in technical_indicators):
            return True
        
        # Check if it's a compound technical term
        if re.search(r'[a-z][A-Z]|[a-z]\+\+|[a-z]#', term):  # CamelCase or special chars
            return True
        
        return False
    
    def extract_experience_years(self, text: str) -> float:
        """Extract maximum experience years mentioned."""
        experience_years = []
        
        for pattern in self.experience_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years = float(match.group(1))
                    if 0 <= years <= 50:  # Reasonable range
                        experience_years.append(years)
                except (ValueError, IndexError):
                    continue
        
        return max(experience_years) if experience_years else 0.0
    
    def extract_education_level(self, text: str) -> str:
        """Extract highest education level with better accuracy."""
        text_lower = text.lower()
        
        education_scores = {'diploma': 0, 'bachelor': 0, 'master': 0, 'phd': 0}
        
        for level_index, pattern in enumerate(self.education_patterns):
            matches = re.findall(pattern, text_lower)
            if matches:
                level_names = ['diploma', 'bachelor', 'master', 'phd']
                education_scores[level_names[level_index]] += len(matches)
        
        # Return the highest level found
        max_level = max(education_scores.items(), key=lambda x: x[1])
        return max_level[0] if max_level[1] > 0 else 'unknown'

class DocumentToDocumentMatcher:
    """Main class for matching two documents (JD and Resume)."""
    
    def __init__(self):
        self.parser = DocumentParser()
        self.extractor = EnhancedSkillExtractor()
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=2000, min_df=1)
    
    def analyze_documents(self, jd_file_path: str, resume_file_path: str) -> HardScoreResult:
        """Main method to analyze JD and Resume documents."""
        
        try:
            # Extract text from both documents
            jd_text = self.parser.extract_text(jd_file_path)
            resume_text = self.parser.extract_text(resume_file_path)
            
            if not jd_text.strip():
                raise ValueError("Could not extract text from job description file")
            
            if not resume_text.strip():
                raise ValueError("Could not extract text from resume file")
            
            # Analyze both documents
            jd_analysis = self.extractor.analyze_document(jd_text)
            resume_analysis = self.extractor.analyze_document(resume_text)
            
            logger.info(f"JD Skills: {sum(len(skills) for skills in jd_analysis.skills.values())}")
            logger.info(f"Resume Skills: {sum(len(skills) for skills in resume_analysis.skills.values())}")
            
            # Calculate matching scores
            skill_matches, missing_skills = self._match_skills(resume_analysis.skills, jd_analysis.skills, resume_text, jd_text)
            technical_score = self._calculate_technical_skills_score(skill_matches, jd_analysis.skills)
            qualification_score = self._calculate_qualification_score(resume_analysis.education_level, jd_analysis.education_level)
            experience_score = self._calculate_experience_score(resume_analysis.experience_years, jd_analysis.experience_years)
            keyword_density_score = self._calculate_keyword_density_score(resume_text, jd_text)
            requirement_match_score = self._calculate_requirement_match_score(resume_text, jd_analysis.key_requirements)
            
            # Calculate weighted overall score
            weights = {
                'technical_skills': 0.35,
                'qualifications': 0.15,
                'experience': 0.15,
                'keyword_density': 0.20,
                'requirement_match': 0.15
            }
            
            overall_score = (
                technical_score * weights['technical_skills'] +
                qualification_score * weights['qualifications'] +
                experience_score * weights['experience'] +
                keyword_density_score * weights['keyword_density'] +
                requirement_match_score * weights['requirement_match']
            )
            
            # Determine verdict and generate suggestions
            verdict = self._determine_verdict(overall_score)
            suggestions = self._generate_suggestions(missing_skills, overall_score, skill_matches, jd_analysis, resume_analysis)
            
            return HardScoreResult(
                overall_score=round(overall_score, 2),
                skill_matches=skill_matches,
                missing_skills=missing_skills,
                technical_skills_score=round(technical_score, 2),
                qualification_score=round(qualification_score, 2),
                experience_score=round(experience_score, 2),
                keyword_density_score=round(keyword_density_score, 2),
                requirement_match_score=round(requirement_match_score, 2),
                breakdown={
                    'technical_skills': round(technical_score, 2),
                    'qualifications': round(qualification_score, 2),
                    'experience': round(experience_score, 2),
                    'keyword_density': round(keyword_density_score, 2),
                    'requirement_match': round(requirement_match_score, 2)
                },
                verdict=verdict,
                suggestions=suggestions,
                jd_analysis=jd_analysis,
                resume_analysis=resume_analysis
            )
            
        except Exception as e:
            logger.error(f"Error analyzing documents: {e}")
            return self._create_error_result(str(e))
    
    def _match_skills(self, resume_skills: Dict, jd_skills: Dict, resume_text: str, jd_text: str) -> Tuple[List[SkillMatch], List[str]]:
        """Enhanced skill matching with category weighting."""
        matches = []
        missing_skills = []
        
        # Flatten skills with category context
        resume_all_skills = set()
        jd_all_skills = set()
        
        skill_category_map = {}
        
        for category, skills_set in resume_skills.items():
            for skill in skills_set:
                resume_all_skills.add(skill)
                skill_category_map[skill] = category
        
        for category, skills_set in jd_skills.items():
            for skill in skills_set:
                jd_all_skills.add(skill)
                skill_category_map[skill] = category
        
        # Exact matches
        exact_matches = resume_all_skills.intersection(jd_all_skills)
        for skill in exact_matches:
            matches.append(SkillMatch(
                skill=skill,
                match_type='exact',
                confidence=1.0,
                matched_text=skill,
                jd_context=self._get_context(jd_text, skill),
                resume_context=self._get_context(resume_text, skill)
            ))
        
        # Fuzzy matches for remaining JD skills
        remaining_jd_skills = jd_all_skills - exact_matches
        remaining_resume_skills = resume_all_skills - exact_matches
        
        for jd_skill in remaining_jd_skills:
            # Find best fuzzy match
            if remaining_resume_skills:
                best_match = process.extractOne(
                    jd_skill, 
                    list(remaining_resume_skills), 
                    scorer=fuzz.token_sort_ratio
                )
                
                if best_match and best_match[1] >= 75:  # 75% similarity threshold
                    matches.append(SkillMatch(
                        skill=jd_skill,
                        match_type='fuzzy',
                        confidence=best_match[1] / 100.0,
                        matched_text=best_match[0],
                        jd_context=self._get_context(jd_text, jd_skill),
                        resume_context=self._get_context(resume_text, best_match[0])
                    ))
                    remaining_resume_skills.discard(best_match[0])
                else:
                    missing_skills.append(jd_skill)
            else:
                missing_skills.append(jd_skill)
        
        return matches, missing_skills
    
    def _get_context(self, text: str, skill: str, context_length: int = 80) -> str:
        """Get context around a skill mention with better formatting."""
        text_lower = text.lower()
        skill_lower = skill.lower()
        
        # Try exact match first
        index = text_lower.find(skill_lower)
        if index == -1:
            # Try word boundary match
            pattern = r'\b' + re.escape(skill_lower) + r'\b'
            match = re.search(pattern, text_lower)
            if match:
                index = match.start()
            else:
                return f"[{skill} mentioned]"
        
        start = max(0, index - context_length)
        end = min(len(text), index + len(skill) + context_length)
        
        context = text[start:end].strip()
        # Clean up context
        context = re.sub(r'\s+', ' ', context)
        
        return f"...{context}..." if start > 0 or end < len(text) else context
    
    def _calculate_technical_skills_score(self, skill_matches: List[SkillMatch], jd_skills: Dict) -> float:
        """Calculate technical skills score with category weighting."""
        if not jd_skills:
            return 0.0
        
        # Category weights (more important categories have higher weights)
        category_weights = {
            'programming_languages': 1.2,
            'frameworks_libraries': 1.1,
            'databases': 1.0,
            'cloud_devops': 0.9,
            'tools_software': 0.8,
            'methodologies_concepts': 0.7,
            'soft_skills': 0.5,
            'technical_terms': 0.6,
            'entities': 0.5
        }
        
        total_weighted_jd_skills = 0
        matched_weighted_skills = 0
        
        # Calculate weighted totals
        for category, skills in jd_skills.items():
            weight = category_weights.get(category, 0.5)
            total_weighted_jd_skills += len(skills) * weight
        
        for match in skill_matches:
            # Find category for the matched skill
            for category, skills in jd_skills.items():
                if match.skill in skills:
                    weight = category_weights.get(category, 0.5)
                    matched_weighted_skills += match.confidence * weight
                    break
        
        if total_weighted_jd_skills == 0:
            return 0.0
        
        score = (matched_weighted_skills / total_weighted_jd_skills) * 100
        return min(100.0, score)
    
    def _calculate_qualification_score(self, resume_education: str, jd_education: str) -> float:
        """Calculate qualification matching score."""
        education_hierarchy = {
            'unknown': 0,
            'diploma': 1,
            'bachelor': 2,
            'master': 3,
            'phd': 4
        }
        
        resume_level = education_hierarchy.get(resume_education, 0)
        jd_level = education_hierarchy.get(jd_education, 0)
        
        if jd_level == 0:  # No specific requirement in JD
            return 75.0 if resume_level > 0 else 50.0
        
        if resume_level >= jd_level:
            return 100.0
        elif resume_level == jd_level - 1:
            return 80.0
        elif resume_level > 0:
            return 60.0
        else:
            return 30.0
    
    def _calculate_experience_score(self, resume_exp: float, jd_exp: float) -> float:
        """Calculate experience matching score with nuanced evaluation."""
        if jd_exp == 0:  # No specific requirement in JD
            return 80.0 if resume_exp > 0 else 60.0
        
        if resume_exp >= jd_exp:
            # Bonus for exceeding requirements, but cap it
            bonus = min(10, (resume_exp - jd_exp) * 2)
            return min(100.0, 100.0 + bonus)
        elif resume_exp >= jd_exp * 0.8:  # 80% of required experience
            return 85.0
        elif resume_exp >= jd_exp * 0.6:  # 60% of required experience
            return 70.0
        elif resume_exp >= jd_exp * 0.4:  # 40% of required experience
            return 55.0
        else:
            return max(20.0, (resume_exp / jd_exp) * 50)
    
    def _calculate_keyword_density_score(self, resume_text: str, jd_text: str) -> float:
        """Calculate keyword density using advanced TF-IDF analysis."""
        try:
            # Preprocess texts
            documents = [resume_text.lower(), jd_text.lower()]
            
            # Remove very common words and short words
            processed_docs = []
            for doc in documents:
                # Remove punctuation and normalize
                doc = re.sub(r'[^\w\s]', ' ', doc)
                doc = re.sub(r'\s+', ' ', doc)
                processed_docs.append(doc)
            
            # Fit TF-IDF
            tfidf_matrix = self.tfidf.fit_transform(processed_docs)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Convert to percentage and apply scaling
            score = similarity * 100
            
            # Apply non-linear scaling to make differences more pronounced
            if score > 50:
                score = 50 + (score - 50) * 1.5
            
            return min(100.0, score)
            
        except Exception as e:
            logger.warning(f"Error calculating keyword density: {e}")
            return 0.0
    
    def _calculate_requirement_match_score(self, resume_text: str, requirements: List[str]) -> float:
        """Calculate how well the resume matches specific job requirements."""
        if not requirements:
            return 75.0  # Default score if no requirements found
        
        resume_lower = resume_text.lower()
        matched_requirements = 0
        partial_matches = 0
        
        for requirement in requirements:
            requirement_lower = requirement.lower()
            
            # Extract key terms from requirement
            key_terms = self._extract_key_terms_from_requirement(requirement_lower)
            
            if not key_terms:
                continue
            
            # Check for exact matches
            exact_match_count = sum(1 for term in key_terms if term in resume_lower)
            
            if exact_match_count == len(key_terms):
                matched_requirements += 1
            elif exact_match_count >= len(key_terms) * 0.6:  # 60% of terms match
                partial_matches += 0.7
            elif exact_match_count > 0:
                partial_matches += 0.3
        
        total_matches = matched_requirements + partial_matches
        match_ratio = total_matches / len(requirements)
        
        return min(100.0, match_ratio * 100)
    
    def _extract_key_terms_from_requirement(self, requirement: str) -> List[str]:
        """Extract key searchable terms from a requirement string."""
        # Remove common requirement language
        stop_phrases = [
            'experience in', 'knowledge of', 'familiarity with', 'working with',
            'understanding of', 'ability to', 'skills in', 'proficiency in',
            'years of', 'minimum', 'required', 'must have', 'should have',
            'preferred', 'plus', 'bonus', 'nice to have'
        ]
        
        cleaned_req = requirement
        for phrase in stop_phrases:
            cleaned_req = re.sub(phrase, ' ', cleaned_req, flags=re.IGNORECASE)
        
        # Extract meaningful terms (avoid very common words)
        words = re.findall(r'\b\w{3,}\b', cleaned_req)
        
        common_words = {
            'and', 'or', 'the', 'with', 'for', 'in', 'on', 'at', 'to', 'from',
            'by', 'of', 'as', 'is', 'are', 'was', 'were', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
            'may', 'might', 'must', 'can', 'shall', 'work', 'working', 'development',
            'using', 'use', 'able', 'strong', 'good', 'excellent', 'team', 'project'
        }
        
        key_terms = [word.lower() for word in words if word.lower() not in common_words and len(word) > 2]
        
        # Return unique terms, limited to avoid noise
        return list(set(key_terms))[:8]
    
    def _determine_verdict(self, overall_score: float) -> str:
        """Determine verdict with more nuanced thresholds."""
        if overall_score >= 85:
            return "High"
        elif overall_score >= 65:
            return "Medium"
        else:
            return "Low"
    
    def _generate_suggestions(self, missing_skills: List[str], overall_score: float, 
                           matches: List[SkillMatch], jd_analysis: DocumentAnalysis, 
                           resume_analysis: DocumentAnalysis) -> List[str]:
        """Generate detailed improvement suggestions."""
        suggestions = []
        
        # Skills-based suggestions
        if missing_skills:
            # Prioritize missing skills by category importance
            critical_missing = [skill for skill in missing_skills[:8]]
            if critical_missing:
                suggestions.append(f"Add these key missing skills: {', '.join(critical_missing)}")
        
        # Experience suggestions
        if jd_analysis.experience_years > resume_analysis.experience_years:
            exp_gap = jd_analysis.experience_years - resume_analysis.experience_years
            suggestions.append(f"Consider highlighting relevant experience (job requires {jd_analysis.experience_years} years, resume shows {resume_analysis.experience_years} years)")
        
        # Education suggestions
        education_hierarchy = {'unknown': 0, 'diploma': 1, 'bachelor': 2, 'master': 3, 'phd': 4}
        if education_hierarchy.get(jd_analysis.education_level, 0) > education_hierarchy.get(resume_analysis.education_level, 0):
            suggestions.append(f"Consider pursuing {jd_analysis.education_level} degree as mentioned in job requirements")
        
        # Fuzzy match suggestions
        fuzzy_matches = [m for m in matches if m.match_type == 'fuzzy' and m.confidence < 0.9]
        if fuzzy_matches and len(fuzzy_matches) > 2:
            suggestions.append("Use exact terminology from job description to improve keyword matching")
        
        # Content density suggestions
        if overall_score < 70:
            if len(resume_analysis.skills) < len(jd_analysis.skills):
                suggestions.append("Expand technical skills section with more relevant technologies")
            suggestions.append("Include more projects demonstrating the required skills")
        
        # Requirement-specific suggestions
        if jd_analysis.key_requirements:
            unaddressed_reqs = []
            resume_lower = resume_analysis.raw_text.lower()
            
            for req in jd_analysis.key_requirements[:5]:  # Check top 5 requirements
                req_lower = req.lower()
                key_terms = self._extract_key_terms_from_requirement(req_lower)
                
                if key_terms:
                    match_count = sum(1 for term in key_terms if term in resume_lower)
                    if match_count < len(key_terms) * 0.5:  # Less than 50% match
                        unaddressed_reqs.append(req[:60] + "..." if len(req) > 60 else req)
            
            if unaddressed_reqs:
                suggestions.append(f"Address these job requirements: {'; '.join(unaddressed_reqs[:2])}")
        
        # Score-based suggestions
        if overall_score >= 85:
            suggestions.append("Excellent match! Consider emphasizing matching skills in cover letter")
        elif overall_score >= 65:
            suggestions.append("Good match with room for improvement in highlighted areas")
        else:
            suggestions.append("Consider gaining experience in the missing skills before applying")
        
        return suggestions[:6]  # Limit to 6 most important suggestions
    
    def _create_error_result(self, error_message: str) -> HardScoreResult:
        """Create error result for exception cases."""
        return HardScoreResult(
            overall_score=0.0,
            skill_matches=[],
            missing_skills=[],
            technical_skills_score=0.0,
            qualification_score=0.0,
            experience_score=0.0,
            keyword_density_score=0.0,
            requirement_match_score=0.0,
            breakdown={},
            verdict="Low",
            suggestions=[f"Error: {error_message}"],
            jd_analysis=DocumentAnalysis("", {}, 0.0, "unknown", [], []),
            resume_analysis=DocumentAnalysis("", {}, 0.0, "unknown", [], [])
        )

# Main function and usage example
def main():
    """Example usage of the Document-to-Document matcher."""
    
    # Initialize the matcher
    matcher = DocumentToDocumentMatcher()
    
    # File paths (replace with your actual file paths)
    jd_file_path = "sample_jd_2.pdf"  # or .docx
    resume_file_path = "resume - 3.pdf"       # or .docx
    
    try:
        print("Analyzing documents...")
        result = matcher.analyze_documents(jd_file_path, resume_file_path)
        
        # Print comprehensive results
        print("\n" + "="*60)
        print("RESUME-JD MATCHING ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nOVERALL SCORE: {result.overall_score}/100")
        print(f"VERDICT: {result.verdict}")
        
        print(f"\nSCORE BREAKDOWN:")
        print(f"├─ Technical Skills: {result.technical_skills_score}/100")
        print(f"├─ Qualifications: {result.qualification_score}/100") 
        print(f"├─ Experience: {result.experience_score}/100")
        print(f"├─ Keyword Density: {result.keyword_density_score}/100")
        print(f"└─ Requirement Match: {result.requirement_match_score}/100")
        
        print(f"\nDOCUMENT ANALYSIS:")
        print(f"JD Experience Required: {result.jd_analysis.experience_years} years")
        print(f"Resume Experience: {result.resume_analysis.experience_years} years")
        print(f"JD Education Level: {result.jd_analysis.education_level.title()}")
        print(f"Resume Education Level: {result.resume_analysis.education_level.title()}")
        
        print(f"\nSKILL MATCHES ({len(result.skill_matches)}):")
        for i, match in enumerate(result.skill_matches[:10], 1):
            print(f"{i:2d}. {match.skill} ({match.match_type}, {match.confidence:.2f})")
            if match.matched_text != match.skill:
                print(f"    └─ Matched as: {match.matched_text}")
        
        if len(result.skill_matches) > 10:
            print(f"    ... and {len(result.skill_matches) - 10} more matches")
        
        print(f"\nMISSING SKILLS ({len(result.missing_skills)}):")
        for i, skill in enumerate(result.missing_skills[:10], 1):
            print(f"{i:2d}. {skill}")
        
        if len(result.missing_skills) > 10:
            print(f"    ... and {len(result.missing_skills) - 10} more missing skills")
        
        print(f"\nJOB REQUIREMENTS ({len(result.jd_analysis.key_requirements)}):")
        for i, req in enumerate(result.jd_analysis.key_requirements[:5], 1):
            print(f"{i}. {req[:80]}{'...' if len(req) > 80 else ''}")
        
        print(f"\nIMPROVEMENT SUGGESTIONS:")
        for i, suggestion in enumerate(result.suggestions, 1):
            print(f"{i}. {suggestion}")
        
        print("\n" + "="*60)
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please ensure both job description and resume files exist.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        logger.exception("Detailed error information:")

def analyze_files(jd_path: str, resume_path: str) -> HardScoreResult:
    """Convenience function to analyze two files and return results."""
    matcher = DocumentToDocumentMatcher()
    return matcher.analyze_documents(jd_path, resume_path)

if __name__ == "__main__":
    main()