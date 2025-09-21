# RESUME-ANALYZER
This is an automated relevance resume checker using NLP Task Processing as well as Hard Score Semantics where job recruiters can update their job desciptions in pdf/docx format.
Students/Candidates can also upload multiple resumes in pdf/docx format.
The system checks for the relevance of resume from the job desciption and provides a score in two forms:Hard Score and Semnatic Score.
Then the final Hybrid Score is calculated using both of them.

This system also consists of an AI Performace Enhancer which gives suggesions on improvizations which can be made from the resume like skills and project recommendations.
The system also consists of a management system for recruiters which can tell them about the score traccking of multiple resumes and help them sort them to make the process of hiring easier.
The recruiters can easily sort the resumes by storing the results in a database as the system is also flexible in that.

The system is conneted to a very simple web app application by connecting it to a flask based server and an html web page which allows for multiple inputs and the system runs on two backend services running parellel.
The two backend services include hard score calculation from the resume which is semi dynamic in nature since it requires a list of words to itertate and some sort of learning is also made through basic learnings.
The second is a pure dynamic LLM -semantic and embedding based score calculation which uses two LLM models one for text generation and one for embeddings which calculates the score.

It also includes a AI Based Recommendation/FeedBack System which reviews the Resume and provides insights for it and also recommends if there are changes to be made.
Finally all the dependancies are connected via LangChain Systems and are available in Requirements.Txt which includes the steps and techstack used.
