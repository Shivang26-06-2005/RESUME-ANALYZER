
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
        """Perform fuzzy matching of skills"""
        if not required_skills:
            return [], 0.0
            
        matched_skills = []
        total_score = 0
        
        for req_skill in required_skills:
            best_match_score = 0
            best_match = None
            
            for resume_skill in resume_skills:
                score = fuzz.ratio(req_skill.lower(), resume_skill.lower())
                if score > best_match_score:
                    best_match_score = score
                    best_match = resume_skill
            
            if best_match_score > 70:  # Threshold for fuzzy match
                matched_skills.append(req_skill)
                total_score += best_match_score
        
        avg_score = total_score / len(required_skills) if required_skills else 0
        return matched_skills, avg_score / 100.0

class SemanticMatcher:
    """Handle semantic matching using embeddings and LLM"""
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        # Use Sentence Transformers for embeddings (works independently of Ollama)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Using HuggingFace all-MiniLM-L6-v2 for embeddings")
        except ImportError:
            logger.error("HuggingFaceEmbeddings not available. Install with: pip install sentence-transformers")
            self.embeddings = None
        except Exception as e:
            logger.error(f"Could not load embedding model: {e}")
            self.embeddings = None
        
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
    
    def calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculate semantic similarity using Ollama embeddings and scale to 25 points"""
        if not self.embeddings:
            logger.warning("No embedding model available, returning 0.0 semantic similarity")
            return 0.0
            
        try:
            # Get embeddings for both texts (reduced size for smaller models)
            resume_embedding = self.embeddings.embed_query(resume_text[:800])  
            jd_embedding = self.embeddings.embed_query(jd_text[:800])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                np.array(resume_embedding).reshape(1, -1),
                np.array(jd_embedding).reshape(1, -1)
            )
            
            # Scale the similarity score (which is between -1 and 1) to be between 0 and 25
            scaled_score = (float(similarity[0][0]) + 1) * 12.5  # Convert from [-1,1] to [0,25]
            return round(scaled_score, 2)  # Round to 2 decimal places
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0  # Return 0 score on error
            # Use smaller context for embedding
            jd_embedding = self.embeddings.embed_query(jd_text[:800])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                np.array(resume_embedding).reshape(1, -1),
                np.array(jd_embedding).reshape(1, -1)
            )
            
            return float(similarity[0][0])
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def generate_feedback_and_suggestions(self, resume: ParsedResume, jd: ParsedJD, score: float) -> Tuple[Dict[str, List[str]], List[str]]:
        """Generate missing items and improvement suggestions using LangChain"""
        
        try:
            response = self.feedback_chain.invoke({