import spacy
import re
import pymupdf  # fitz
import docx2txt
from typing import Dict, List, Set
from collections import defaultdict, Counter
from fuzzywuzzy import fuzz
import os
from pathlib import Path
import logging
from dataclasses import dataclass

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
    verdict: str
    suggestions: List[str]
    jd_analysis: DocumentAnalysis
    resume_analysis: DocumentAnalysis

# ---------------- Document Parser ----------------
class DocumentParser:
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        try:
            doc = pymupdf.open(file_path)
            text = ""
            for page in doc:
                page_text = page.get_text()
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
        try:
            text = docx2txt.process(file_path)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""

    @staticmethod
    def extract_text(file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        file_extension = Path(file_path).suffix.lower()
        if file_extension == '.pdf':
            return DocumentParser.extract_text_from_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return DocumentParser.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported: .pdf, .docx, .doc")

# ---------------- Job Description Parser ----------------
class JobDescriptionParser:
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
        requirements = []
        text_lower = text.lower()
        for pattern in self.requirement_indicators:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start_pos = match.end()
                section_end = min(len(text), start_pos + 1000)
                section_text = text[start_pos:section_end]
                req_items = self._parse_bullet_points(section_text)
                requirements.extend(req_items)
        if not requirements:
            requirements = self._extract_implicit_requirements(text)
        return list(set(requirements))

    def extract_responsibilities(self, text: str) -> List[str]:
        responsibilities = []
        text_lower = text.lower()
        for pattern in self.responsibility_indicators:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start_pos = match.end()
                section_end = min(len(text), start_pos + 1000)
                section_text = text[start_pos:section_end]
                resp_items = self._parse_bullet_points(section_text)
                responsibilities.extend(resp_items)
        return list(set(responsibilities))

    def _parse_bullet_points(self, text: str) -> List[str]:
        items = []
        bullet_patterns = [
            r'(?:^|\n)\s*[-•*]\s*(.+?)(?=\n\s*[-•*]|\n\s*\n|\Z)',
            r'(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\n\s*\n|\Z)',
            r'(?:^|\n)\s*[a-zA-Z]\)\s*(.+?)(?=\n\s*[a-zA-Z]\)|\n\s*\n|\Z)',
        ]
        for pattern in bullet_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                item = match.group(1).strip()
                if len(item) > 10:
                    items.append(item)
        if not items:
            sentences = re.split(r'[.!?]\s+', text)
            items = [s.strip() for s in sentences if len(s.strip()) > 20]
        return items[:10]

    def _extract_implicit_requirements(self, text: str) -> List[str]:
        requirements = []
        exp_matches = re.findall(r'(\d+[\+\-]*\s*years?\s+(?:of\s+)?experience[^.]*)', text, re.IGNORECASE)
        requirements.extend(exp_matches)
        degree_matches = re.findall(r'((?:bachelor|master|phd)[^.]*)', text, re.IGNORECASE)
        requirements.extend(degree_matches)
        return requirements[:15]

# ---------------- Enhanced Skill Extractor ----------------
class EnhancedSkillExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        self.experience_patterns = [
            r'(\d+(?:\.\d+)?)[\+\-\s]*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp|work)',
        ]
        self.education_patterns = [
            r'\b(?:bachelor|b\.?tech|b\.?sc|bcom|ba|bs|undergraduate)\b',
            r'\b(?:master|m\.?tech|m\.?sc|mcom|ma|ms|graduate)\b',
            r'\b(?:phd|ph\.?d|doctorate|doctoral|postgraduate)\b'
        ]

    def analyze_document(self, text: str) -> DocumentAnalysis:
        skills = self.extract_skills_from_text(text)
        experience_years = self.extract_experience_years(text)
        education_level = self.extract_education_level(text)
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
        extracted_skills = defaultdict(set)
        text_lower = text.lower()
        if self.nlp:
            doc = self.nlp(text)
            for chunk in doc.noun_chunks:
                term = chunk.text.lower().strip()
                if self._is_technical_term(term):
                    extracted_skills['technical_terms'].add(term)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'GPE'] and self._is_technical_term(ent.text.lower()):
                    extracted_skills['entities'].add(ent.text.lower())
        words = re.findall(r'\b\w{2,}\b', text_lower)
        counter = Counter(words)
        for word, _ in counter.most_common(150):
            if self._is_technical_term(word):
                extracted_skills['technical_terms'].add(word)
        return {k: v for k, v in extracted_skills.items() if v}

    def _is_technical_term(self, term: str) -> bool:
        if len(term) < 2 or len(term) > 50:
            return False
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as'}
        if term.lower() in common_words:
            return False
        return True

    def extract_experience_years(self, text: str) -> float:
        experience_years = []
        for pattern in self.experience_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years = float(match.group(1))
                    if 0 <= years <= 50:
                        experience_years.append(years)
                except:
                    continue
        return max(experience_years) if experience_years else 0.0

    def extract_education_level(self, text: str) -> str:
        text_lower = text.lower()
        education_scores = {'bachelor': 0, 'master': 0, 'phd': 0}
        for idx, pattern in enumerate(self.education_patterns):
            matches = re.findall(pattern, text_lower)
            if matches:
                keys = ['bachelor', 'master', 'phd']
                education_scores[keys[idx]] += len(matches)
        max_level = max(education_scores.items(), key=lambda x: x[1])
        return max_level[0] if max_level[1] > 0 else 'unknown'

# ---------------- Matcher and Scoring ----------------
class DocumentMatcher:
    def __init__(self, jd_analysis: DocumentAnalysis, resume_analysis: DocumentAnalysis):
        self.jd = jd_analysis
        self.resume = resume_analysis

    def match_skills(self) -> List[SkillMatch]:
        matches = []
        jd_skills = set()
        for sset in self.jd.skills.values():
            jd_skills.update(sset)
        resume_skills = set()
        for sset in self.resume.skills.values():
            resume_skills.update(sset)
        for skill in jd_skills:
            if skill in resume_skills:
                matches.append(SkillMatch(skill, 'exact', 100.0, skill, skill, skill))
            else:
                # fuzzy matching
                best = None
                best_score = 0
                for r_skill in resume_skills:
                    score = fuzz.partial_ratio(skill, r_skill)
                    if score > best_score:
                        best_score = score
                        best = r_skill
                if best_score >= 80:
                    matches.append(SkillMatch(skill, 'fuzzy', best_score, best, skill, best))
        return matches

    def compute_scores(self) -> HardScoreResult:
        skill_matches = self.match_skills()
        jd_skills = set()
        for sset in self.jd.skills.values():
            jd_skills.update(sset)
        matched_skills = {sm.skill for sm in skill_matches}
        missing_skills = list(jd_skills - matched_skills)
        technical_skills_score = min(100.0, len(matched_skills) / max(1, len(jd_skills)) * 100)
        experience_score = min(100.0, self.resume.experience_years / max(1, self.jd.experience_years) * 100)
        qualification_score = 100.0 if self.resume.education_level == self.jd.education_level else 50.0
        requirement_match_score = 100.0 if len(missing_skills) == 0 else max(0.0, 100.0 - len(missing_skills) * 10)
        overall_score = (technical_skills_score*0.4 + experience_score*0.3 + qualification_score*0.2 + requirement_match_score*0.1)
        verdict = 'High' if overall_score >= 80 else 'Medium' if overall_score >= 50 else 'Low'
        suggestions = []
        if missing_skills:
            suggestions.append(f"Consider improving skills: {', '.join(missing_skills)}")
        return HardScoreResult(
            overall_score=overall_score,
            skill_matches=skill_matches,
            missing_skills=missing_skills,
            technical_skills_score=technical_skills_score,
            qualification_score=qualification_score,
            experience_score=experience_score,
            keyword_density_score=0,
            requirement_match_score=requirement_match_score,
            breakdown={'technical_skills_score': technical_skills_score,
                       'experience_score': experience_score,
                       'qualification_score': qualification_score,
                       'requirement_match_score': requirement_match_score},
            verdict=verdict,
            suggestions=suggestions,
            jd_analysis=self.jd,
            resume_analysis=self.resume
        )

# ---------------- Main ----------------
def main(jd_file: str, resume_file: str):
    parser = DocumentParser()
    jd_text = parser.extract_text(jd_file)
    resume_text = parser.extract_text(resume_file)

    skill_extractor = EnhancedSkillExtractor()
    jd_analysis = skill_extractor.analyze_document(jd_text)
    resume_analysis = skill_extractor.analyze_document(resume_text)

    matcher = DocumentMatcher(jd_analysis, resume_analysis)
    result = matcher.compute_scores()

    print("Overall Score:", result.overall_score)
    print("Verdict:", result.verdict)
    print("Skill Matches:")
    for sm in result.skill_matches:
        print(f" - {sm.skill} ({sm.match_type}, confidence={sm.confidence})")
    if result.missing_skills:
        print("Missing Skills:", result.missing_skills)
    if result.suggestions:
        print("Suggestions:", result.suggestions)

# ---------------- Run ----------------
if __name__ == "__main__":
    # Replace with your actual file paths
    jd_file_path = "sample_jd_2.pdf"
    resume_file_path = "resume - 3.pdf"
    main(jd_file_path, resume_file_path)
