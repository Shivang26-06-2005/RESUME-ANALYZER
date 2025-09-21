from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import tempfile
import logging
from typing import Dict, List, Any
import traceback
from dataclasses import asdict
import json

# Import our matching systems
# Note: You'll need to modify the imports based on your file structure
from Semantics_2 import ResumeJDMatcher, BatchProcessor, MatchingResult
from Hard_Matching import DocumentToDocumentMatcher, HardScoreResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

class HybridMatcher:
    """Combines semantic and hard matching approaches"""
    
    def __init__(self):
        # Initialize semantic matcher with mxbai-embed-large as primary
        self.semantic_matcher = ResumeJDMatcher(
            model_name="gemma3:1b", 
            langsmith_api_key=None
        )
        # Update embedding model priority to use mxbai-embed-large
        self._update_embedding_priority()
        
        self.hard_matcher = DocumentToDocumentMatcher()
        self.batch_processor = BatchProcessor(self.semantic_matcher)
        
        # Hybrid scoring weights
        self.weights = {
            'semantic_weight': 0.4,
            'hard_weight': 0.6
        }
    
    def _update_embedding_priority(self):
        """Update the embedding model priority to use mxbai-embed-large as primary"""
        # Modify the semantic matcher's embedding initialization
        model_variants = [
            "mxbai-embed-large",  # Primary choice as specified
            "nomic-embed-text",   # Alternative 1
            "all-MiniLM-L6-v2"    # Fallback
        ]
        
        # Update the semantic matcher's embedding model priority
        if hasattr(self.semantic_matcher, 'semantic_matcher'):
            original_init = self.semantic_matcher.semantic_matcher._initialize_embeddings
            
            def updated_init():
                from langchain_ollama import OllamaEmbeddings
                
                for model_name in model_variants:
                    try:
                        logger.info(f"Attempting to initialize embedding model: {model_name}")
                        embeddings = OllamaEmbeddings(
                            model=model_name,
                            base_url="http://localhost:11434"
                        )
                        
                        # Test the embedding model
                        test_embedding = embeddings.embed_query("test embedding functionality")
                        
                        if test_embedding and len(test_embedding) > 0:
                            logger.info(f"Successfully initialized {model_name} for embeddings")
                            self.semantic_matcher.semantic_matcher.embeddings = embeddings
                            return
                            
                    except Exception as e:
                        logger.warning(f"Could not initialize {model_name}: {e}")
                        continue
                
                # Call original initialization as fallback
                original_init()
            
            # Replace the initialization method
            self.semantic_matcher.semantic_matcher._initialize_embeddings = updated_init
    
    def analyze_multiple_resumes(self, jd_path: str, resume_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple resumes against a job description using hybrid approach"""
        results = []
        analytics = {
            'total_resumes': len(resume_paths),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_score': 0.0,
            'high_matches': 0,
            'medium_matches': 0,
            'low_matches': 0,
            'semantic_scores': [],
            'hard_scores': []
        }
        
        for resume_path in resume_paths:
            try:
                # Get semantic analysis
                semantic_result = self.semantic_matcher.match_resume_to_jd(resume_path, jd_path)
                
                # Get hard matching analysis
                hard_result = self.hard_matcher.analyze_documents(jd_path, resume_path)
                
                # Calculate hybrid score
                hybrid_score = self._calculate_hybrid_score(semantic_result, hard_result)
                
                # Extract resume name from filename
                resume_name = os.path.splitext(os.path.basename(resume_path))[0]
                
                # Combine results
                combined_result = {
                    'filename': os.path.basename(resume_path),
                    'resume_name': resume_name,
                    'hybrid_score': hybrid_score,
                    'verdict': self._determine_hybrid_verdict(hybrid_score),
                    
                    # Semantic scores
                    'semantic_score': semantic_result.semantic_similarity,
                    'semantic_relevance': semantic_result.relevance_score,
                    
                    # Hard matching scores
                    'hard_score': hard_result.overall_score,
                    'technical_score': hard_result.technical_skills_score,
                    'qualification_score': hard_result.qualification_score,
                    'experience_score': hard_result.experience_score,
                    'keyword_density_score': hard_result.keyword_density_score,
                    'requirement_match_score': hard_result.requirement_match_score,
                    
                    # Skills analysis
                    'matched_skills': [match.skill for match in hard_result.skill_matches],
                    'missing_skills': hard_result.missing_skills + semantic_result.missing_skills,
                    
                    # AI suggestions (combine both sources and keep all suggestions)
                    'suggestions': self._combine_and_deduplicate_suggestions(
                        semantic_result.suggestions, 
                        hard_result.suggestions
                    )
                }
                
                results.append(combined_result)
                
                # Update analytics
                analytics['successful_analyses'] += 1
                analytics['semantic_scores'].append(semantic_result.semantic_similarity)
                analytics['hard_scores'].append(hard_result.overall_score)
                
                verdict = combined_result['verdict'].lower()
                if verdict == 'high':
                    analytics['high_matches'] += 1
                elif verdict == 'medium':
                    analytics['medium_matches'] += 1
                else:
                    analytics['low_matches'] += 1
                    
            except Exception as e:
                logger.error(f"Error analyzing {resume_path}: {e}")
                analytics['failed_analyses'] += 1
                
                results.append({
                    'filename': os.path.basename(resume_path),
                    'resume_name': os.path.splitext(os.path.basename(resume_path))[0],
                    'error': str(e),
                    'hybrid_score': 0.0,
                    'verdict': 'Error'
                })
        
        # Calculate final analytics
        if results:
            valid_scores = [r['hybrid_score'] for r in results if 'error' not in r]
            analytics['average_score'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        # Sort results by hybrid score
        results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        return {
            'results': results,
            'analytics': analytics
        }
    
    def _calculate_hybrid_score(self, semantic_result: MatchingResult, hard_result: HardScoreResult) -> float:
        """Calculate hybrid score combining semantic and hard matching"""
        semantic_score = semantic_result.semantic_similarity * 100  # Convert to percentage
        hard_score = hard_result.overall_score
        
        # Weighted combination
        hybrid_score = (
            semantic_score * self.weights['semantic_weight'] +
            hard_score * self.weights['hard_weight']
        )
        
        return min(100.0, hybrid_score)
    
    def _combine_and_deduplicate_suggestions(self, semantic_suggestions: List[str], 
                                           hard_suggestions: List[str]) -> List[str]:
        """Combine suggestions from both sources, deduplicate, and prioritize"""
        # Combine all suggestions
        all_suggestions = semantic_suggestions + hard_suggestions
        
        # Remove duplicates while preserving order and similarity
        unique_suggestions = []
        seen_keywords = set()
        
        for suggestion in all_suggestions:
            # Extract key words for similarity checking
            key_words = set(suggestion.lower().split()[:5])  # First 5 words
            
            # Check if this suggestion is too similar to existing ones
            is_duplicate = any(
                len(key_words.intersection(existing_keywords)) >= 3 
                for existing_keywords in seen_keywords
            )
            
            if not is_duplicate and suggestion.strip():
                unique_suggestions.append(suggestion.strip())
                seen_keywords.add(frozenset(key_words))
        
        # Limit to reasonable number but don't truncate too aggressively
        return unique_suggestions[:10]
    
    def _determine_hybrid_verdict(self, hybrid_score: float) -> str:
        """Determine verdict based on hybrid score"""
        if hybrid_score >= 80:
            return "High"
        elif hybrid_score >= 60:
            return "Medium"
        else:
            return "Low"

# Initialize the hybrid matcher
hybrid_matcher = HybridMatcher()

@app.route('/')
def index():
    """Serve the frontend HTML"""
    # Read the HTML content from the artifact
    # In production, you would save the HTML as a template file
    with open('index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return render_template_string(html_content)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Main endpoint for document analysis"""
    try:
        # Validate request
        if 'jd_file' not in request.files:
            return jsonify({'error': 'Job description file is required'}), 400
        
        if 'resume_files' not in request.files:
            return jsonify({'error': 'At least one resume file is required'}), 400
        
        jd_file = request.files['jd_file']
        resume_files = request.files.getlist('resume_files')
        
        if not jd_file.filename:
            return jsonify({'error': 'Job description file is empty'}), 400
        
        if not resume_files or not any(f.filename for f in resume_files):
            return jsonify({'error': 'No resume files provided'}), 400
        
        # Validate file types
        allowed_extensions = {'.pdf', '.docx', '.doc'}
        
        def validate_file(file):
            if not file.filename:
                return False, "Empty filename"
            
            ext = os.path.splitext(file.filename.lower())[1]
            if ext not in allowed_extensions:
                return False, f"Invalid file type: {ext}"
            
            return True, "Valid"
        
        # Validate JD file
        is_valid, error = validate_file(jd_file)
        if not is_valid:
            return jsonify({'error': f'Job description file error: {error}'}), 400
        
        # Validate resume files
        for resume_file in resume_files:
            is_valid, error = validate_file(resume_file)
            if not is_valid:
                return jsonify({'error': f'Resume file error: {error}'}), 400
        
        # Save files temporarily
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save JD file
            jd_filename = os.path.join(temp_dir, f"jd_{jd_file.filename}")
            jd_file.save(jd_filename)
            
            # Save resume files
            resume_paths = []
            for i, resume_file in enumerate(resume_files):
                resume_filename = os.path.join(temp_dir, f"resume_{i}_{resume_file.filename}")
                resume_file.save(resume_filename)
                resume_paths.append(resume_filename)
            
            # Analyze documents
            logger.info(f"Starting analysis with JD: {jd_file.filename} and {len(resume_paths)} resumes")
            
            analysis_result = hybrid_matcher.analyze_multiple_resumes(jd_filename, resume_paths)
            
            logger.info(f"Analysis completed. {analysis_result['analytics']['successful_analyses']} successful analyses")
            
            return jsonify(analysis_result)
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        
        finally:
            # Cleanup temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp files: {cleanup_error}")
    
    except Exception as e:
        logger.error(f"Unexpected error in analyze endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if Ollama is accessible
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model="gemma3:1b", timeout=5)
        
        # Try a simple generation
        response = llm.invoke("Hello")
        
        return jsonify({
            'status': 'healthy',
            'ollama_status': 'connected',
            'models': {
                'llm': 'gemma3:1b',
                'embedding': 'mxbai-embed-large'
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'message': 'Check if Ollama is running and models are installed'
        }), 503

@app.route('/models', methods=['GET'])
def list_models():
    """List available Ollama models"""
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({
                'models': result.stdout,
                'status': 'success'
            })
        else:
            return jsonify({
                'error': result.stderr,
                'status': 'error'
            }), 500
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Could not list Ollama models'
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Resume-JD Matching System")
    print("🔧 Configuration:")
    print(f"   - LLM Model: gemma3:1b")
    print(f"   - Embedding Model: mxbai-embed-large (primary)")
    print(f"   - Max file size: 50MB")
    print(f"   - Temp directory: {tempfile.gettempdir()}")
    print("")
    print("📋 Prerequisites:")
    print("   1. Ollama server running: ollama serve")
    print("   2. Models installed:")
    print("      - ollama pull gemma3:1b")
    print("      - ollama pull mxbai-embed-large")
    print("   3. Python dependencies installed")
    print("")
    print("🌐 Endpoints:")
    print("   - /: Frontend interface")
    print("   - /analyze: Document analysis (POST)")
    print("   - /health: Health check")
    print("   - /models: List Ollama models")
    print("")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)