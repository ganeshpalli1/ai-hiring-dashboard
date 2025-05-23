# Resume Screening System with Azure OpenAI - Simplified Version
# Production-grade application for analyzing resumes with classification

import os
import asyncio
import logging
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import time
from functools import wraps
import re
from collections import defaultdict
import uuid
from io import BytesIO
import traceback

import aiofiles
from pydantic import BaseModel, Field, field_validator
from openai import AzureOpenAI
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import PyPDF2
import docx
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Response, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
import uvicorn

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Please install with: pip install python-dotenv")
    print("   Environment variables will be read from system environment")

# Configuration
class Config:
    """Application configuration"""
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    
    MAX_TOKENS_PER_REQUEST = 2000
    MAX_RETRIES = 3
    BATCH_SIZE = 50
    MAX_CONCURRENT_REQUESTS = 10
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls):
        """Validate that required environment variables are set"""
        errors = []
        if not cls.AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY is not set")
        if not cls.AZURE_OPENAI_ENDPOINT:
            errors.append("AZURE_OPENAI_ENDPOINT is not set")
        
        if errors:
            print("❌ Environment Variable Errors:")
            for error in errors:
                print(f"   - {error}")
            print("\n📝 Please check your .env file or system environment variables")
            return False
        else:
            print("✅ All required environment variables are set")
            print(f"   • Endpoint: {cls.AZURE_OPENAI_ENDPOINT}")
            print(f"   • Deployment: {cls.AZURE_OPENAI_DEPLOYMENT}")
            print(f"   • API Version: {cls.AZURE_OPENAI_API_VERSION}")
            return True

# Initialize logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize metrics - handle duplicates gracefully
try:
    resume_processed_counter = Counter('resumes_processed_total', 'Total number of resumes processed')
except ValueError:
    # Metric already exists, create a dummy counter that does nothing
    class DummyCounter:
        def inc(self, amount=1): pass
        def labels(self, **kwargs): return self
    resume_processed_counter = DummyCounter()

try:
    processing_time_histogram = Histogram('resume_processing_duration_seconds', 'Resume processing duration')
except ValueError:
    class DummyHistogram:
        def time(self): 
            return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    processing_time_histogram = DummyHistogram()

try:
    active_jobs_gauge = Gauge('active_processing_jobs', 'Number of active processing jobs')
except ValueError:
    class DummyGauge:
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
        def set(self, value): pass
        @property
        def _value(self):
            class DummyValue:
                def get(self): return 0
            return DummyValue()
    active_jobs_gauge = DummyGauge()

try:
    classification_counter = Counter('resume_classification', 'Resume classifications', ['category', 'level'])
except ValueError:
    class DummyLabeledCounter:
        def labels(self, **kwargs): 
            class DummyCounter:
                def inc(self, amount=1): pass
            return DummyCounter()
    classification_counter = DummyLabeledCounter()

# In-memory storage
class InMemoryStore:
    """Simple in-memory storage for jobs and results"""
    def __init__(self):
        self.jobs = {}
        self.resume_analyses = defaultdict(list)
        self.processing_status = defaultdict(lambda: {"total": 0, "processed": 0})
    
    def create_job(self, job_id: str, job_data: Dict[str, Any]):
        """Store job data"""
        self.jobs[job_id] = {
            **job_data,
            "created_at": datetime.utcnow().isoformat(),
            "analysis": None
        }
    
    def update_job_analysis(self, job_id: str, analysis: Dict[str, Any]):
        """Update job analysis"""
        if job_id in self.jobs:
            self.jobs[job_id]["analysis"] = analysis
    
    def add_resume_analysis(self, job_id: str, analysis: Dict[str, Any]):
        """Add resume analysis result"""
        self.resume_analyses[job_id].append(analysis)
        self.processing_status[job_id]["processed"] += 1
    
    def increment_total_resumes(self, job_id: str, count: int):
        """Increment total resume count"""
        self.processing_status[job_id]["total"] += count
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job data"""
        return self.jobs.get(job_id)
    
    def get_results(self, job_id: str, min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get results for a job"""
        results = self.resume_analyses.get(job_id, [])
        if min_score:
            results = [r for r in results if r.get("fit_score", 0) >= min_score]
        return sorted(results, key=lambda x: x.get("fit_score", 0), reverse=True)
    
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get processing status"""
        return self.processing_status.get(job_id, {"total": 0, "processed": 0})

# Initialize storage
storage = InMemoryStore()

# Pydantic Models
class JobDescriptionInput(BaseModel):
    job_role: str = Field(..., min_length=1, max_length=255)
    required_experience: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10, max_length=10000)
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if len(v.split()) < 10:
            raise ValueError('Job description must contain at least 10 words')
        return v

class ResumeClassification(BaseModel):
    category: str  # tech, non-tech, semi-tech
    level: str     # entry, mid, senior
    confidence: float

class ResumeAnalysisResult(BaseModel):
    resume_id: str
    filename: str
    classification: ResumeClassification
    fit_score: float
    matching_skills: List[str]
    missing_skills: List[str]
    recommendation: str
    detailed_analysis: Dict[str, Any]

# Azure OpenAI Client
class AzureOpenAIClient:
    """Wrapper for Azure OpenAI with rate limiting and error handling"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
        )
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.rate_limiter = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def complete(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """Make completion request with retry logic"""
        async with self.rate_limiter:
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=Config.AZURE_OPENAI_DEPLOYMENT,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=Config.MAX_TOKENS_PER_REQUEST
                )
                
                # Extract content and validate
                content = response.choices[0].message.content
                if content is None:
                    logger.error("OpenAI returned null content")
                    raise ValueError("OpenAI returned null content")
                
                # Log response details for debugging
                logger.info(f"OpenAI response received - Content length: {len(content)}")
                logger.debug(f"Response content preview: {content[:100] if content else 'EMPTY'}...")
                
                return content
                
            except Exception as e:
                logger.error(f"Azure OpenAI API error: {str(e)}")
                logger.error(f"Error details: {traceback.format_exc()}")
                raise

# Resume Parser
class ResumeParser:
    """Extract text from various resume formats"""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF"""
        try:
            # Wrap bytes in BytesIO for PyPDF2
            file_stream = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(file_stream)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}")
            raise
    
    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """Extract text from DOCX"""
        try:
            # Wrap bytes in BytesIO for python-docx
            file_stream = BytesIO(file_content)
            doc = docx.Document(file_stream)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting DOCX: {str(e)}")
            raise
    
    @classmethod
    def extract_text(cls, file_content: bytes, filename: str) -> str:
        """Extract text based on file type"""
        if filename.lower().endswith('.pdf'):
            return cls.extract_text_from_pdf(file_content)
        elif filename.lower().endswith(('.docx', '.doc')):
            return cls.extract_text_from_docx(file_content)
        elif filename.lower().endswith('.txt'):
            return file_content.decode('utf-8', errors='ignore')
        else:
            raise ValueError(f"Unsupported file format: {filename}")

# Job Analyzer
class JobAnalyzer:
    """Analyze job descriptions using LLM"""
    
    def __init__(self, openai_client: AzureOpenAIClient):
        self.openai_client = openai_client
    
    async def analyze_job_description(self, job_role: str, required_experience: str, 
                                     description: str) -> Dict[str, Any]:
        """Extract skills and analyze job description"""
        
        prompt = f"""
        Analyze the following job description and extract key information:
        
        Job Role: {job_role}
        Required Experience: {required_experience}
        Description: {description}
        
        Please provide a comprehensive analysis in JSON format with the following structure:
        {{
            "required_skills": {{
                "technical": ["list of technical skills"],
                "soft": ["list of soft skills"],
                "domain": ["domain-specific skills"]
            }},
            "nice_to_have_skills": ["optional skills"],
            "key_responsibilities": ["main responsibilities"],
            "required_qualifications": ["education, certifications, etc."],
            "experience_requirements": {{
                "years": "extracted years of experience",
                "type": "type of experience needed"
            }},
            "technology_stack": ["specific technologies mentioned"],
            "industry_domain": "identified industry/domain",
            "job_category": "tech/non-tech/semi-tech classification"
        }}
        
        Be thorough and extract both explicit and implicit requirements.
        Respond ONLY with valid JSON, no additional text or formatting.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert HR analyst specializing in job requirement extraction. You must respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.openai_client.complete(messages)
            
            # Log the raw response for debugging
            logger.info(f"Raw OpenAI response length: {len(response) if response else 0}")
            if not response:
                logger.error("Empty response from OpenAI")
                raise ValueError("Empty response from OpenAI")
            
            # Clean the response - remove any markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            logger.info(f"Cleaned response preview: {cleaned_response[:200]}...")
            
            try:
                analysis = json.loads(cleaned_response)
                logger.info("Successfully parsed job analysis JSON")
                return analysis
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                logger.error(f"Attempted to parse: {cleaned_response[:500]}...")
                
                # Fallback: create a basic analysis structure
                fallback_analysis = {
                    "required_skills": {
                        "technical": ["Programming", "Software Development"],
                        "soft": ["Communication", "Problem Solving"],
                        "domain": ["Business Requirements"]
                    },
                    "nice_to_have_skills": ["Team Leadership"],
                    "key_responsibilities": ["Develop software solutions", "Collaborate with team"],
                    "required_qualifications": ["Bachelor's degree", "Relevant experience"],
                    "experience_requirements": {
                        "years": required_experience,
                        "type": "Professional experience"
                    },
                    "technology_stack": ["General technology stack"],
                    "industry_domain": "Technology",
                    "job_category": "tech"
                }
                logger.warning("Using fallback analysis due to JSON parsing error")
                return fallback_analysis
                
        except Exception as e:
            logger.error(f"Job analysis error: {str(e)}")
            logger.error(f"Full error details: {traceback.format_exc()}")
            
            # Return a basic fallback analysis instead of raising
            fallback_analysis = {
                "required_skills": {
                    "technical": ["Programming", "Software Development"],
                    "soft": ["Communication", "Problem Solving"],
                    "domain": ["Business Requirements"]
                },
                "nice_to_have_skills": ["Team Leadership"],
                "key_responsibilities": ["Develop software solutions", "Collaborate with team"],
                "required_qualifications": ["Bachelor's degree", "Relevant experience"],
                "experience_requirements": {
                    "years": required_experience,
                    "type": "Professional experience"
                },
                "technology_stack": ["General technology stack"],
                "industry_domain": "Technology",
                "job_category": "tech"
            }
            logger.warning("Using fallback analysis due to API error")
            return fallback_analysis

# Resume Analyzer with Classification
class ResumeAnalyzer:
    """Analyze and classify resumes against job descriptions"""
    
    def __init__(self, openai_client: AzureOpenAIClient):
        self.openai_client = openai_client
    
    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair common JSON issues"""
        try:
            logger.info(f"Starting JSON repair for {len(json_str)} character response")
            
            # Start with the original string
            repaired = json_str.strip()
            
            # Remove any markdown formatting that might have been missed
            if repaired.startswith('```'):
                lines = repaired.split('\n')
                # Find first line that starts with {
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('{'):
                        start_idx = i
                        break
                # Find last line that ends with }
                end_idx = len(lines) - 1
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().endswith('}'):
                        end_idx = i
                        break
                repaired = '\n'.join(lines[start_idx:end_idx + 1])
            
            # Fix common JSON issues step by step
            
            # 1. Remove trailing commas before closing braces/brackets
            repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
            
            # 2. Fix unescaped quotes in strings (basic approach)
            # Look for patterns like "text with "quotes" inside"
            repaired = re.sub(r'"([^"]*)"([^"]*)"([^"]*)"', r'"\1\\"2\\"\3"', repaired)
            
            # 3. Handle incomplete strings at the end
            # If there's an unmatched quote at the end, close it
            quote_count = repaired.count('"')
            if quote_count % 2 != 0:
                # Find the last quote and see if it needs closing
                last_quote_idx = repaired.rfind('"')
                if last_quote_idx > 0:
                    # Look for the pattern: "key": "incomplete_value
                    after_quote = repaired[last_quote_idx + 1:]
                    if not after_quote.strip().endswith('"') and not after_quote.strip().endswith('"}'):
                        # Add closing quote
                        repaired = repaired[:last_quote_idx + 1] + after_quote.split('\n')[0].strip() + '"'
                        # Remove any text after that
                        lines = repaired.split('\n')
                        repaired = lines[0] if len(lines) > 1 else repaired
            
            # 4. Ensure proper structure completion
            # Count opening and closing braces/brackets
            open_braces = repaired.count('{')
            close_braces = repaired.count('}')
            open_brackets = repaired.count('[')
            close_brackets = repaired.count(']')
            
            # Add missing closing braces and brackets
            while close_braces < open_braces:
                repaired += '}'
                close_braces += 1
            while close_brackets < open_brackets:
                repaired += ']'
                close_brackets += 1
            
            # 5. Remove any trailing text after the last complete JSON object
            # Find the last properly closed brace
            brace_count = 0
            last_valid_pos = -1
            for i, char in enumerate(repaired):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_pos = i
            
            if last_valid_pos > 0 and last_valid_pos < len(repaired) - 1:
                repaired = repaired[:last_valid_pos + 1]
            
            # 6. Handle missing commas between key-value pairs
            # Look for patterns like: "key1": "value1" "key2": "value2"
            repaired = re.sub(r'"\s*"\s*([a-zA-Z_][a-zA-Z0-9_]*)":', r'", "\1":', repaired)
            
            # 7. Fix missing commas after array/object elements
            # Pattern: } "key": becomes }, "key":
            repaired = re.sub(r'}\s*"([^"]+)":', r'}, "\1":', repaired)
            # Pattern: ] "key": becomes ], "key":
            repaired = re.sub(r']\s*"([^"]+)":', r'], "\1":', repaired)
            
            logger.info(f"JSON repair completed. Original: {len(json_str)}, Repaired: {len(repaired)}")
            logger.debug(f"Repaired JSON preview: {repaired[:300]}...")
            
            return repaired
            
        except Exception as e:
            logger.error(f"Error during JSON repair: {str(e)}")
            logger.error(f"Full repair error: {traceback.format_exc()}")
            return json_str  # Return original if repair fails
    
    async def classify_resume(self, resume_text: str) -> ResumeClassification:
        """Classify resume into category and level"""
        
        prompt = f"""
        Classify the following resume into appropriate categories:
        
        RESUME:
        {resume_text}
        
        Provide classification in JSON format:
        {{
            "category": "tech/non-tech/semi-tech",
            "level": "entry/mid/senior",
            "confidence": 0.0-1.0,
            "reasoning": {{
                "category_reasoning": "explanation for category classification",
                "level_reasoning": "explanation for level classification",
                "key_indicators": ["list of key indicators used for classification"]
            }}
        }}
        
        Category definitions:
        - tech: Primarily technical roles (developers, engineers, data scientists, etc.)
        - non-tech: Non-technical roles (HR, sales, marketing, operations, etc.)
        - semi-tech: Mixed technical and non-technical (technical PM, business analyst, etc.)
        
        Level definitions:
        - entry: 0-2 years experience or fresh graduate
        - mid: 3-7 years experience
        - senior: 8+ years experience or leadership roles
        
        Consider education, years of experience, job titles, skills, and responsibilities.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert resume classifier with deep understanding of various industries and roles. IMPORTANT: You must respond with valid, well-formatted JSON only. Do not include any text before or after the JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.openai_client.complete(messages, temperature=0.1)
            
            # Log the raw response for debugging  
            logger.info(f"Classification response length: {len(response) if response else 0}")
            if not response:
                logger.error("Empty classification response from OpenAI")
                raise ValueError("Empty classification response from OpenAI")
            
            # Clean the response - remove any markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            logger.info(f"Cleaned classification response preview: {cleaned_response[:200]}...")
            
            try:
                classification_data = json.loads(cleaned_response)
                logger.info("Successfully parsed classification JSON")
            except json.JSONDecodeError as e:
                logger.error(f"Classification JSON decode error: {str(e)}")
                logger.error(f"Attempted to parse: {cleaned_response[:500]}...")
                
                # Fallback classification
                classification_data = {
                    "category": "tech",
                    "level": "mid", 
                    "confidence": 0.5
                }
                logger.warning("Using fallback classification due to JSON parsing error")
            
            # Track classification metrics
            classification_counter.labels(
                category=classification_data["category"],
                level=classification_data["level"]
            ).inc()
            
            return ResumeClassification(
                category=classification_data["category"],
                level=classification_data["level"],
                confidence=classification_data["confidence"]
            )
        except Exception as e:
            logger.error(f"Resume classification error: {str(e)}")
            logger.error(f"Full error details: {traceback.format_exc()}")
            # Default classification on error
            return ResumeClassification(
                category="tech",
                level="mid",
                confidence=0.5
            )
    
    async def analyze_resume(self, resume_text: str, job_analysis: Dict[str, Any], 
                           job_description: str, classification: ResumeClassification) -> Dict[str, Any]:
        """Analyze resume fit for job with classification context"""
        
        prompt = f"""
        Analyze the following resume against the job requirements:
        
        RESUME CLASSIFICATION:
        - Category: {classification.category}
        - Level: {classification.level}
        
        JOB REQUIREMENTS:
        {json.dumps(job_analysis, indent=2)}
        
        ORIGINAL JOB DESCRIPTION:
        {job_description}
        
        RESUME:
        {resume_text}
        
        Provide analysis in this EXACT JSON format (no additional text, no markdown):
        {{
            "fit_score": 0-100,
            "matching_skills": ["skill1", "skill2", "skill3"],
            "missing_skills": ["missing1", "missing2"],
            "experience_score": 0-100,
            "recommendation": "STRONG_FIT or GOOD_FIT or MODERATE_FIT or WEAK_FIT",
            "detailed_feedback": "Single paragraph comprehensive feedback"
        }}
        
        Keep it simple and ensure valid JSON syntax.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert technical recruiter with deep understanding of skill assessment, resume analysis, and role-level matching. IMPORTANT: You must respond with valid, well-formatted JSON only. Do not include any text before or after the JSON. Ensure all strings are properly quoted and escaped, and all nested structures are complete."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.openai_client.complete(messages, temperature=0.2)
            
            # Log the raw response for debugging
            logger.info(f"Analysis response length: {len(response) if response else 0}")
            if not response:
                logger.error("Empty analysis response from OpenAI")
                raise ValueError("Empty analysis response from OpenAI")
            
            # Clean the response - remove any markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            logger.info(f"Cleaned analysis response preview: {cleaned_response[:200]}...")
            
            try:
                analysis = json.loads(cleaned_response)
                logger.info("Successfully parsed analysis JSON")
                return analysis
            except json.JSONDecodeError as e:
                logger.error(f"Analysis JSON decode error: {str(e)}")
                logger.error(f"Attempted to parse: {cleaned_response[:500]}...")
                
                # Try to repair common JSON issues
                try:
                    logger.info("Attempting JSON repair...")
                    repaired_json = self._repair_json(cleaned_response)
                    analysis = json.loads(repaired_json)
                    logger.info("Successfully parsed repaired JSON")
                    return analysis
                except Exception as repair_error:
                    logger.error(f"JSON repair failed: {str(repair_error)}")
                
                # Create fallback analysis
                fallback_analysis = {
                    "fit_score": 50.0,
                    "matching_skills": ["Analysis failed"],
                    "missing_skills": ["Manual review required"],
                    "experience_score": 50,
                    "recommendation": "MANUAL_REVIEW",
                    "detailed_feedback": "Automatic analysis failed due to parsing error. Manual review recommended."
                }
                logger.warning("Using fallback analysis due to JSON parsing error")
                return fallback_analysis
                
        except json.JSONDecodeError:
            logger.error("Failed to parse resume analysis response")
            raise
        except Exception as e:
            logger.error(f"Resume analysis error: {str(e)}")
            logger.error(f"Full error details: {traceback.format_exc()}")
            raise

# Continue with remaining classes and application code...
# [The file is quite long, so I'll create it in parts]

if __name__ == "__main__":
    # Validate environment variables
    if not Config.validate():
        print("\n💡 To fix this, create a .env file in the project root with:")
        print("   AZURE_OPENAI_API_KEY=your_api_key_here")
        print("   AZURE_OPENAI_ENDPOINT=your_endpoint_here")
        print("   AZURE_OPENAI_DEPLOYMENT=your_deployment_name (optional, defaults to gpt-4)")
        print("   LOG_LEVEL=INFO (optional)")
        exit(1)
    
    uvicorn.run(
        "resumematching:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker since we're using in-memory storage
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": Config.LOG_LEVEL,
                "handlers": ["default"],
            },
        }
    )