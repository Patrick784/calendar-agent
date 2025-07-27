"""
Security and Privacy Module

Provides PII sanitization, input validation, and secure logging functionality
to protect user data throughout the calendar agent system.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import hashlib
import json
from datetime import datetime

# PII detection patterns
PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone": re.compile(r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
    "ssn": re.compile(r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
    "zip_code": re.compile(r'\b\d{5}(?:-\d{4})?\b'),
    "name_patterns": [
        re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),  # First Last
        re.compile(r'\b[A-Z]\. [A-Z][a-z]+\b'),      # J. Smith
        re.compile(r'\b[A-Z][a-z]+, [A-Z][a-z]+\b'), # Smith, John
    ],
    "address": re.compile(r'\b\d+\s+[A-Za-z0-9\s,]+(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Boulevard|Blvd)\b', re.IGNORECASE),
}

# Sensitive keywords that should be redacted
SENSITIVE_KEYWORDS = {
    "password", "passwd", "pwd", "secret", "token", "key", "api_key", 
    "oauth", "credential", "auth", "private", "confidential", "salary",
    "income", "medical", "health", "diagnosis", "prescription", "ssn",
    "social_security", "credit_card", "bank_account", "routing_number"
}

# Field names that commonly contain PII
PII_FIELD_NAMES = {
    "email", "phone", "telephone", "mobile", "address", "street", "zip",
    "postal_code", "ssn", "social_security_number", "credit_card", "cc",
    "bank_account", "routing", "first_name", "last_name", "full_name",
    "name", "contact", "emergency_contact", "next_of_kin"
}

class PIISanitizer:
    """
    PII sanitization utility for protecting user data.
    
    Provides methods to detect, redact, and hash PII in text and structured data.
    """
    
    def __init__(self, 
                 hash_salt: str = None, 
                 preserve_format: bool = True,
                 redaction_char: str = "*"):
        """
        Initialize PII sanitizer.
        
        Args:
            hash_salt: Salt for hashing PII (uses default if None)
            preserve_format: Whether to preserve original format when redacting
            redaction_char: Character to use for redaction
        """
        self.hash_salt = hash_salt or "calendar_agent_default_salt_2024"
        self.preserve_format = preserve_format
        self.redaction_char = redaction_char
        self.logger = logging.getLogger("security.pii_sanitizer")
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitize input text by removing/redacting PII.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text with PII redacted
        """
        if not text or not isinstance(text, str):
            return text
        
        sanitized = text
        
        # Email addresses
        sanitized = self._redact_pattern(sanitized, PII_PATTERNS["email"], "EMAIL")
        
        # Phone numbers
        sanitized = self._redact_pattern(sanitized, PII_PATTERNS["phone"], "PHONE")
        
        # SSN
        sanitized = self._redact_pattern(sanitized, PII_PATTERNS["ssn"], "SSN")
        
        # Credit cards
        sanitized = self._redact_pattern(sanitized, PII_PATTERNS["credit_card"], "CREDIT_CARD")
        
        # IP addresses
        sanitized = self._redact_pattern(sanitized, PII_PATTERNS["ip_address"], "IP_ADDRESS")
        
        # ZIP codes (be careful not to redact legitimate postal codes in addresses)
        sanitized = self._redact_pattern(sanitized, PII_PATTERNS["zip_code"], "ZIP_CODE")
        
        # Addresses
        sanitized = self._redact_pattern(sanitized, PII_PATTERNS["address"], "ADDRESS")
        
        # Names (be conservative to avoid false positives)
        for name_pattern in PII_PATTERNS["name_patterns"]:
            sanitized = self._redact_pattern(sanitized, name_pattern, "NAME")
        
        # Sensitive keywords
        sanitized = self._redact_sensitive_keywords(sanitized)
        
        return sanitized
    
    def sanitize_dict(self, data: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
        """
        Sanitize a dictionary by removing PII from values and sensitive field names.
        
        Args:
            data: Dictionary to sanitize
            deep: Whether to recursively sanitize nested dictionaries
            
        Returns:
            Sanitized dictionary
        """
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        
        for key, value in data.items():
            # Check if field name is sensitive
            if key.lower() in PII_FIELD_NAMES:
                # Hash or redact the value
                if isinstance(value, str):
                    sanitized[key] = self._hash_value(value)
                else:
                    sanitized[key] = "[REDACTED]"  
                continue
            
            # Sanitize the value
            if isinstance(value, str):
                sanitized[key] = self.sanitize_input(value)
            elif isinstance(value, dict) and deep:
                sanitized[key] = self.sanitize_dict(value, deep)
            elif isinstance(value, list) and deep:
                sanitized[key] = [
                    self.sanitize_dict(item, deep) if isinstance(item, dict)
                    else self.sanitize_input(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def detect_pii(self, text: str) -> List[Dict[str, str]]:
        """
        Detect PII in text without redacting it.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected PII items with type and location
        """
        if not text or not isinstance(text, str):
            return []
        
        detected = []
        
        # Check each PII pattern
        for pii_type, pattern in PII_PATTERNS.items():
            if pii_type == "name_patterns":
                for i, name_pattern in enumerate(pattern):
                    matches = name_pattern.finditer(text)
                    for match in matches:
                        detected.append({
                            "type": f"name_pattern_{i}",
                            "text": match.group(),
                            "start": match.start(),
                            "end": match.end()
                        })
            else:
                matches = pattern.finditer(text)
                for match in matches:
                    detected.append({
                        "type": pii_type,
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end()
                    })
        
        return detected
    
    def _redact_pattern(self, text: str, pattern: re.Pattern, label: str) -> str:
        """Redact matches of a regex pattern"""
        
        def replace_match(match):
            original = match.group()
            if self.preserve_format:
                # Preserve some structure (e.g., keep @ in emails, dashes in phones)
                if label == "EMAIL":
                    parts = original.split("@")
                    if len(parts) == 2:
                        return f"[EMAIL]@{parts[1]}"
                elif label == "PHONE":
                    # Keep formatting but redact numbers
                    redacted = re.sub(r'\d', self.redaction_char, original)
                    return f"[PHONE:{redacted}]"
                elif label == "SSN":
                    return "***-**-****"
                elif label == "CREDIT_CARD":
                    return "****-****-****-****"
            
            return f"[{label}]"
        
        return pattern.sub(replace_match, text)
    
    def _redact_sensitive_keywords(self, text: str) -> str:
        """Redact sensitive keywords"""
        
        words = text.split()
        sanitized_words = []
        
        for word in words:
            # Remove punctuation for checking
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if clean_word in SENSITIVE_KEYWORDS:
                # Replace with redacted version preserving length
                redacted = self.redaction_char * len(word)
                sanitized_words.append(f"[{redacted}]")
            else:
                sanitized_words.append(word)
        
        return " ".join(sanitized_words)
    
    def _hash_value(self, value: str) -> str:
        """Hash a value with salt for privacy while maintaining searchability"""
        
        if not value:
            return value
        
        # Create a hash that's consistent but not reversible
        salted_value = f"{value}{self.hash_salt}"
        hash_object = hashlib.sha256(salted_value.encode())
        hash_hex = hash_object.hexdigest()
        
        # Return first 16 characters for readability
        return f"hash_{hash_hex[:16]}"
    
    def create_privacy_report(self, text: str) -> Dict[str, Any]:
        """
        Create a privacy report for text showing detected PII.
        
        Args:
            text: Text to analyze
            
        Returns:
            Privacy report with detected PII and risk assessment
        """
        detected_pii = self.detect_pii(text)
        
        # Count types of PII
        pii_counts = {}
        for item in detected_pii:
            pii_type = item["type"]
            pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1
        
        # Assess risk level
        risk_level = "low"
        if len(detected_pii) > 5:
            risk_level = "high"
        elif len(detected_pii) > 2:
            risk_level = "medium"
        
        # Generate recommendations
        recommendations = []
        if "email" in pii_counts:
            recommendations.append("Consider anonymizing email addresses")
        if "phone" in pii_counts:
            recommendations.append("Phone numbers detected - ensure proper consent")
        if "ssn" in pii_counts:
            recommendations.append("SSN detected - high priority for redaction")
        if "credit_card" in pii_counts:
            recommendations.append("Credit card detected - immediately redact")
        
        return {
            "total_pii_items": len(detected_pii),
            "pii_types": list(pii_counts.keys()),
            "pii_counts": pii_counts,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "detected_items": detected_pii
        }

class SecurityLoggingFilter(logging.Filter):
    """
    Logging filter that automatically sanitizes PII from log messages.
    
    Integrates with Python's logging system to ensure no PII is written to logs.
    """
    
    def __init__(self, sanitizer: PIISanitizer = None):
        super().__init__()
        self.sanitizer = sanitizer or PIISanitizer()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter and sanitize log records.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to allow the record, False to block it
        """
        try:
            # Sanitize the main message
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                record.msg = self.sanitizer.sanitize_input(record.msg)
            
            # Sanitize any args
            if hasattr(record, 'args') and record.args:
                sanitized_args = []
                for arg in record.args:
                    if isinstance(arg, str):
                        sanitized_args.append(self.sanitizer.sanitize_input(arg))
                    elif isinstance(arg, dict):
                        sanitized_args.append(self.sanitizer.sanitize_dict(arg))
                    else:
                        sanitized_args.append(arg)
                record.args = tuple(sanitized_args)
            
            # Check for sensitive information in the record itself
            record_dict = record.__dict__
            for key, value in record_dict.items():
                if isinstance(value, str) and key not in ['name', 'levelname', 'filename', 'module']:
                    record_dict[key] = self.sanitizer.sanitize_input(value)
            
            return True
            
        except Exception as e:
            # Don't block logging if sanitization fails, but log the error
            print(f"Logging filter error: {e}")
            return True

def setup_secure_logging(logger_name: str = None, 
                        sanitizer: PIISanitizer = None) -> logging.Logger:
    """
    Set up secure logging with PII filtering.
    
    Args:
        logger_name: Name of logger to configure (uses root if None)
        sanitizer: PII sanitizer to use (creates default if None)
        
    Returns:
        Configured logger with security filters
    """
    sanitizer = sanitizer or PIISanitizer()
    security_filter = SecurityLoggingFilter(sanitizer)
    
    logger = logging.getLogger(logger_name)
    
    # Add security filter to all handlers
    for handler in logger.handlers:
        handler.addFilter(security_filter)
    
    # If no handlers exist, add a basic one with security filter
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(security_filter)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def validate_input_safety(user_input: str, 
                         max_length: int = 10000,
                         allowed_chars: str = None) -> Tuple[bool, List[str]]:
    """
    Validate input for safety and security concerns.
    
    Args:
        user_input: User input to validate
        max_length: Maximum allowed length
        allowed_chars: Set of allowed characters (None for basic validation)
        
    Returns:
        Tuple of (is_safe, list_of_issues)
    """
    issues = []
    
    if not isinstance(user_input, str):
        issues.append("Input must be a string")
        return False, issues
    
    # Length check
    if len(user_input) > max_length:
        issues.append(f"Input exceeds maximum length of {max_length}")
    
    # Check for potential injection attacks
    injection_patterns = [
        r'<script[^>]*>',  # Script tags
        r'javascript:',     # JavaScript URLs
        r'vbscript:',      # VBScript URLs
        r'on\w+\s*=',      # Event handlers
        r'eval\s*\(',      # Eval functions
        r'exec\s*\(',      # Exec functions
        r'system\s*\(',    # System calls
        r'drop\s+table',   # SQL injection
        r'union\s+select', # SQL injection
        r'--\s*$',         # SQL comments
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            issues.append(f"Potential injection detected: {pattern}")
    
    # Character validation
    if allowed_chars:
        invalid_chars = set(user_input) - set(allowed_chars)
        if invalid_chars:
            issues.append(f"Invalid characters detected: {invalid_chars}")
    
    # Check for excessive PII
    sanitizer = PIISanitizer()
    pii_report = sanitizer.create_privacy_report(user_input)
    
    if pii_report["risk_level"] == "high":
        issues.append("High amount of PII detected in input")
    
    return len(issues) == 0, issues

# Default global sanitizer instance
default_sanitizer = PIISanitizer()

# Convenience functions using default sanitizer
def sanitize_input(text: str) -> str:
    """Sanitize input text using default sanitizer"""
    return default_sanitizer.sanitize_input(text)

def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize dictionary using default sanitizer"""
    return default_sanitizer.sanitize_dict(data)

def detect_pii(text: str) -> List[Dict[str, str]]:
    """Detect PII using default sanitizer"""
    return default_sanitizer.detect_pii(text)

def create_privacy_report(text: str) -> Dict[str, Any]:
    """Create privacy report using default sanitizer"""
    return default_sanitizer.create_privacy_report(text) 