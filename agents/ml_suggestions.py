"""
ML Suggestion Agent

Runs ML models (initially TF-IDF + RandomForest) to predict task success 
probabilities and recommend better time slots. Later phases will upgrade 
to vector-based RAG and more sophisticated models, retraining on user feedback.
"""

import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
import logging

# ML and embedding imports with graceful fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available - falling back to TF-IDF")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import pandas as pd
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - ML suggestions disabled")

from .base import BaseAgent, AgentMessage, AgentResponse

# Import validation if available
try:
    from src.validation import validate_task, validate_event, ValidationResult
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

class MLSuggestionAgent(BaseAgent):
    """
    ML-powered suggestion agent for calendar optimization.
    
    Features:
    - Predict success probability of scheduled events
    - Recommend optimal time slots based on historical data
    - Learn from user feedback and scheduling patterns
    - Provide contextual scheduling insights
    - Continuously improve recommendations through retraining
    """
    
    def __init__(self, settings: Dict[str, Any] = None):
        super().__init__("ml_suggestions", settings)
        
        # Model configuration with swappable algorithms
        self.model_type = settings.get("model_type", "rag" if SENTENCE_TRANSFORMERS_AVAILABLE else "tfidf")
        self.model_dir = settings.get("model_dir", "./models")
        
        # Model storage paths
        self.success_model_path = os.path.join(self.model_dir, "success_predictor.pkl")
        self.embedding_model_path = os.path.join(self.model_dir, "embedding_model") 
        self.tfidf_model_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
        self.time_slot_model_path = os.path.join(self.model_dir, "time_slot_predictor.pkl")
        
        # ML models (abstracted interface)
        self.embedding_model = None  # For RAG approach
        self.success_predictor = None
        self.tfidf_vectorizer = None  # Fallback approach
        self.time_slot_predictor = None
        
        # RAG-specific components
        self.knowledge_base: List[Dict[str, Any]] = []
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Training data
        self.training_data: List[Dict[str, Any]] = []
        self.feedback_data: List[Dict[str, Any]] = []
        
        # Model features
        self.feature_columns = [
            'hour_of_day', 'day_of_week', 'month', 'duration_minutes',
            'is_weekend', 'is_work_hours', 'title_length', 'has_attendees',
            'days_in_advance', 'is_recurring'
        ]
        
        # Performance metrics
        self.model_metrics = {
            "model_type": self.model_type,
            "accuracy": 0.0,
            "f1_score": 0.0,
            "last_training": None,
            "training_samples": 0,
            "rag_retrievals": 0 if self.model_type == "rag" else None
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing models
        self._load_models()
        
        self.logger.info(f"ML Suggestion Agent initialized with {self.model_type} model")
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Process ML suggestion requests.
        
        Supported actions:
        - predict_success: Predict success probability for an event
        - recommend_time_slots: Recommend optimal time slots
        - analyze_patterns: Analyze scheduling patterns and insights
        - retrain_models: Retrain models with new feedback data
        """
        
        start_time = datetime.utcnow()
        action = message.body.get("action", "")
        context = message.body.get("context", {})
        
        try:
            if action == "predict_success":
                result = await self._predict_success(context)
            elif action == "recommend_time_slots":
                result = await self._recommend_time_slots(context)
            elif action == "analyze_patterns":
                result = await self._analyze_patterns(context)
            elif action == "retrain_models":
                result = await self._retrain_models(context)
            else:
                return AgentResponse(
                    success=False,
                    error=f"Unknown ML action: {action}"
                )
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.update_metrics(response_time, True)
            
            return AgentResponse(
                success=True,
                data=result,
                metadata={"action": action, "response_time": response_time}
            )
            
        except Exception as e:
            self.logger.error(f"ML suggestion error for action '{action}': {str(e)}")
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.update_metrics(response_time, False)
            
            return AgentResponse(
                success=False,
                error=f"ML prediction failed: {str(e)}"
            )
    
    async def _predict_success(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the success probability of a scheduled event"""
        
        # Extract event details
        event_data = context.get("event_data", {})
        time_slot = context.get("time_slot", {})
        
        if not event_data and not time_slot:
            raise ValueError("Missing event_data or time_slot for prediction")
        
        # Extract features from event data
        features = self._extract_features(event_data, time_slot)
        
        # Make prediction if model is available
        if self.success_predictor and self.tfidf_vectorizer:
            try:
                # Prepare feature vector
                feature_vector = self._prepare_feature_vector(features, event_data)
                
                # Predict success probability
                success_prob = self.success_predictor.predict_proba([feature_vector])[0][1]
                success_prediction = success_prob > 0.5
                
                # Get feature importance for explanation
                feature_importance = dict(zip(
                    self.feature_columns + ['text_features'],
                    list(self.success_predictor.feature_importances_)
                ))
                
                return {
                    "success_probability": float(success_prob),
                    "predicted_success": success_prediction,
                    "confidence": float(max(success_prob, 1 - success_prob)),
                    "feature_importance": feature_importance,
                    "features_used": features,
                    "model_available": True
                }
                
            except Exception as e:
                self.logger.warning(f"ML prediction failed, using heuristics: {str(e)}")
        
        # Fallback to heuristic-based prediction
        heuristic_result = self._heuristic_success_prediction(features, event_data)
        
        return {
            "success_probability": heuristic_result["probability"],
            "predicted_success": heuristic_result["success"],
            "confidence": heuristic_result["confidence"],
            "feature_importance": {},
            "features_used": features,
            "model_available": False,
            "fallback_method": "heuristic"
        }
    
    async def _recommend_time_slots(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal time slots based on ML predictions"""
        
        # Get event requirements
        event_data = context.get("event_data", {})
        candidate_slots = context.get("candidate_slots", [])
        num_recommendations = context.get("num_recommendations", 5)
        
        if not candidate_slots:
            raise ValueError("No candidate time slots provided")
        
        # Score each candidate slot
        scored_slots = []
        
        for slot in candidate_slots:
            # Predict success for this slot
            prediction_context = {
                "event_data": event_data,
                "time_slot": slot
            }
            
            prediction = await self._predict_success(prediction_context)
            
            # Calculate additional scoring factors
            slot_score = self._calculate_slot_score(slot, event_data, prediction)
            
            scored_slots.append({
                **slot,
                "ml_score": prediction["success_probability"],
                "total_score": slot_score["total_score"],
                "score_breakdown": slot_score["breakdown"],
                "predicted_success": prediction["predicted_success"],
                "confidence": prediction["confidence"]
            })
        
        # Sort by total score
        scored_slots.sort(key=lambda x: x["total_score"], reverse=True)
        
        # Return top recommendations
        recommendations = scored_slots[:num_recommendations]
        
        return {
            "recommendations": recommendations,
            "total_candidates_evaluated": len(candidate_slots),
            "scoring_method": "ml_enhanced" if self.success_predictor else "heuristic",
            "recommendation_factors": [
                "success_probability",
                "time_preferences", 
                "schedule_density",
                "historical_patterns"
            ]
        }
    
    async def _analyze_patterns(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scheduling patterns and provide insights"""
        
        # Get historical data
        historical_events = context.get("historical_events", [])
        user_id = context.get("user_id", "default")
        analysis_period = context.get("analysis_period", "month")
        
        if not historical_events:
            return {
                "patterns": {},
                "insights": ["No historical data available for pattern analysis"],
                "recommendations": []
            }
        
        # Analyze scheduling patterns
        patterns = {
            "preferred_hours": self._analyze_time_preferences(historical_events),
            "success_patterns": self._analyze_success_patterns(historical_events),
            "duration_patterns": self._analyze_duration_patterns(historical_events),
            "day_preferences": self._analyze_day_preferences(historical_events),
            "seasonal_trends": self._analyze_seasonal_trends(historical_events)
        }
        
        # Generate insights
        insights = self._generate_insights(patterns, historical_events)
        
        # Generate recommendations based on patterns
        recommendations = self._generate_pattern_recommendations(patterns, insights)
        
        return {
            "patterns": patterns,
            "insights": insights,
            "recommendations": recommendations,
            "analysis_period": analysis_period,
            "events_analyzed": len(historical_events),
            "user_id": user_id
        }
    
    async def _retrain_models(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrain ML models with new feedback data"""
        
        # Get new training data
        new_feedback = context.get("feedback_data", [])
        force_retrain = context.get("force_retrain", False)
        
        # Add new feedback to training data
        self.feedback_data.extend(new_feedback)
        
        # Check if we have enough data for retraining
        min_training_samples = 50
        if len(self.feedback_data) < min_training_samples and not force_retrain:
            return {
                "retrained": False,
                "reason": f"Need at least {min_training_samples} samples for retraining",
                "current_samples": len(self.feedback_data),
                "new_samples_added": len(new_feedback)
            }
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data(self.feedback_data)
            
            if len(X) == 0:
                return {
                    "retrained": False,
                    "reason": "No valid training data after preparation",
                    "samples_processed": len(self.feedback_data)
                }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train new models
            self.success_predictor = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.success_predictor.fit(X_train, y_train)
            
            # Evaluate model performance
            y_pred = self.success_predictor.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save updated models
            self._save_models()
            
            return {
                "retrained": True,
                "model_accuracy": float(accuracy),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "new_samples_added": len(new_feedback),
                "total_feedback_samples": len(self.feedback_data)
            }
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {str(e)}")
            return {
                "retrained": False,
                "reason": f"Retraining failed: {str(e)}",
                "samples_attempted": len(self.feedback_data)
            }
    
    def _extract_features(self, event_data: Dict[str, Any], 
                         time_slot: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from event data and time slot"""
        
        # Get time slot information
        start_time_str = time_slot.get("start_datetime", event_data.get("start_datetime", ""))
        
        if start_time_str:
            try:
                start_time = datetime.fromisoformat(start_time_str.replace('Z', ''))
            except:
                start_time = datetime.utcnow()
        else:
            start_time = datetime.utcnow()
        
        # Extract temporal features
        features = {
            'hour_of_day': start_time.hour,
            'day_of_week': start_time.weekday(),
            'month': start_time.month,
            'is_weekend': start_time.weekday() >= 5,
            'is_work_hours': 9 <= start_time.hour <= 17,
            'days_in_advance': (start_time - datetime.utcnow()).days if start_time > datetime.utcnow() else 0
        }
        
        # Extract event features
        features.update({
            'duration_minutes': time_slot.get("duration_minutes", event_data.get("duration_minutes", 60)),
            'title_length': len(event_data.get("title", "")),
            'has_attendees': len(event_data.get("attendees", [])) > 0,
            'is_recurring': bool(event_data.get("recurrence")),
        })
        
        return features
    
    def _prepare_feature_vector(self, features: Dict[str, Any], 
                               event_data: Dict[str, Any]) -> List[float]:
        """Prepare feature vector for ML model"""
        
        # Numerical features
        feature_vector = [features.get(col, 0) for col in self.feature_columns]
        
        # Text features using TF-IDF
        if self.tfidf_vectorizer:
            title = event_data.get("title", "")
            description = event_data.get("description", "")
            text_content = f"{title} {description}".strip()
            
            if text_content:
                try:
                    tfidf_features = self.tfidf_vectorizer.transform([text_content])
                    # Use average of TF-IDF scores as a single feature
                    text_feature = float(tfidf_features.mean()) if tfidf_features.nnz > 0 else 0.0
                except:
                    text_feature = 0.0
            else:
                text_feature = 0.0
        else:
            text_feature = 0.0
        
        feature_vector.append(text_feature)
        
        return feature_vector
    
    def _heuristic_success_prediction(self, features: Dict[str, Any], 
                                    event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback heuristic-based success prediction"""
        
        score = 0.5  # Base score
        factors = []
        
        # Time-based heuristics
        if features.get('is_work_hours', False):
            score += 0.2
            factors.append("work_hours")
        
        if not features.get('is_weekend', True):
            score += 0.1
            factors.append("weekday")
        
        # Meeting characteristics
        duration = features.get('duration_minutes', 60)
        if 30 <= duration <= 90:  # Reasonable meeting length
            score += 0.15
            factors.append("optimal_duration")
        
        if features.get('days_in_advance', 0) >= 1:  # Planned in advance
            score += 0.1
            factors.append("planned_ahead")
        
        # Cap score at 1.0
        score = min(score, 1.0)
        
        return {
            "probability": score,
            "success": score > 0.6,
            "confidence": 0.7,  # Lower confidence for heuristics
            "factors": factors
        }
    
    def _calculate_slot_score(self, slot: Dict[str, Any], event_data: Dict[str, Any],
                            prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive score for a time slot"""
        
        scores = {
            "ml_prediction": prediction["success_probability"],
            "time_preference": self._score_time_preference(slot),
            "schedule_density": self._score_schedule_density(slot),
            "user_patterns": self._score_user_patterns(slot, event_data)
        }
        
        # Weighted total score
        weights = {
            "ml_prediction": 0.4,
            "time_preference": 0.3,
            "schedule_density": 0.2,
            "user_patterns": 0.1
        }
        
        total_score = sum(scores[factor] * weights[factor] for factor in scores)
        
        return {
            "total_score": total_score,
            "breakdown": scores
        }
    
    def _score_time_preference(self, slot: Dict[str, Any]) -> float:
        """Score based on general time preferences"""
        
        try:
            start_time = datetime.fromisoformat(slot["start_datetime"].replace('Z', ''))
            hour = start_time.hour
            
            # Prefer work hours
            if 9 <= hour <= 17:
                return 0.8
            elif 8 <= hour <= 18:
                return 0.6
            else:
                return 0.3
                
        except:
            return 0.5
    
    def _score_schedule_density(self, slot: Dict[str, Any]) -> float:
        """Score based on schedule density (placeholder)"""
        
        # This would ideally check actual calendar density
        # For now, return neutral score
        return 0.7
    
    def _score_user_patterns(self, slot: Dict[str, Any], event_data: Dict[str, Any]) -> float:
        """Score based on learned user patterns (placeholder)"""
        
        # This would use historical user data
        # For now, return neutral score
        return 0.6
    
    def _prepare_training_data(self, feedback_data: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[int]]:
        """Prepare training data from feedback"""
        
        X = []
        y = []
        
        # Update TF-IDF vectorizer with all text data
        all_texts = []
        for feedback in feedback_data:
            event_data = feedback.get("event_data", {})
            title = event_data.get("title", "")
            description = event_data.get("description", "")
            text_content = f"{title} {description}".strip()
            if text_content:
                all_texts.append(text_content)
        
        if all_texts:
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=100,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            self.tfidf_vectorizer.fit(all_texts)
        
        # Prepare feature vectors
        for feedback in feedback_data:
            try:
                event_data = feedback.get("event_data", {})
                time_slot = feedback.get("time_slot", {})
                success = feedback.get("success", False)
                
                features = self._extract_features(event_data, time_slot)
                feature_vector = self._prepare_feature_vector(features, event_data)
                
                X.append(feature_vector)
                y.append(1 if success else 0)
                
            except Exception as e:
                self.logger.warning(f"Skipping invalid feedback sample: {str(e)}")
                continue
        
        return X, y
    
    def _load_models(self):
        """Load saved ML models"""
        
        try:
            if os.path.exists(self.success_model_path):
                with open(self.success_model_path, 'rb') as f:
                    self.success_predictor = pickle.load(f)
                self.logger.info("Loaded success prediction model")
            
            if os.path.exists(self.tfidf_model_path):
                with open(self.tfidf_model_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                self.logger.info("Loaded TF-IDF vectorizer")
                
        except Exception as e:
            self.logger.warning(f"Failed to load saved models: {str(e)}")
    
    def _save_models(self):
        """Save trained ML models"""
        
        try:
            if self.success_predictor:
                with open(self.success_model_path, 'wb') as f:
                    pickle.dump(self.success_predictor, f)
            
            if self.tfidf_vectorizer:
                with open(self.tfidf_model_path, 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f)
                
            self.logger.info("Saved ML models successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {str(e)}")
    
    def _analyze_time_preferences(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze preferred scheduling times"""
        
        hour_counts = {}
        successful_hours = {}
        
        for event in events:
            try:
                start_time = datetime.fromisoformat(event["start_datetime"].replace('Z', ''))
                hour = start_time.hour
                success = event.get("success", True)  # Assume success if not specified
                
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
                if success:
                    successful_hours[hour] = successful_hours.get(hour, 0) + 1
                    
            except Exception:
                continue
        
        # Find peak hours
        peak_hours = sorted(hour_counts.keys(), key=lambda h: hour_counts[h], reverse=True)[:3]
        
        # Find most successful hours
        success_rates = {}
        for hour in hour_counts:
            if hour_counts[hour] > 0:
                success_rates[hour] = successful_hours.get(hour, 0) / hour_counts[hour]
        
        best_hours = sorted(success_rates.keys(), key=lambda h: success_rates[h], reverse=True)[:3]
        
        return {
            "peak_hours": peak_hours,
            "best_success_hours": best_hours,
            "hour_distribution": hour_counts,
            "success_rates_by_hour": success_rates
        }
    
    def _analyze_success_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze what factors correlate with successful events"""
        
        successful_events = [e for e in events if e.get("success", True)]
        failed_events = [e for e in events if not e.get("success", True)]
        
        return {
            "total_events": len(events),
            "successful_events": len(successful_events),
            "failed_events": len(failed_events),
            "overall_success_rate": len(successful_events) / len(events) if events else 0
        }
    
    def _analyze_duration_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze preferred meeting durations"""
        
        durations = []
        for event in events:
            duration = event.get("duration_minutes", 60)
            durations.append(duration)
        
        if not durations:
            return {"average_duration": 60, "common_durations": []}
        
        avg_duration = sum(durations) / len(durations)
        
        # Find most common durations
        duration_counts = {}
        for duration in durations:
            duration_counts[duration] = duration_counts.get(duration, 0) + 1
        
        common_durations = sorted(duration_counts.keys(), 
                                key=lambda d: duration_counts[d], reverse=True)[:3]
        
        return {
            "average_duration": avg_duration,
            "common_durations": common_durations,
            "duration_distribution": duration_counts
        }
    
    def _analyze_day_preferences(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze preferred days of the week"""
        
        day_counts = {}
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for event in events:
            try:
                start_time = datetime.fromisoformat(event["start_datetime"].replace('Z', ''))
                day = start_time.weekday()
                day_counts[day] = day_counts.get(day, 0) + 1
            except:
                continue
        
        # Convert to day names
        day_name_counts = {day_names[day]: count for day, count in day_counts.items()}
        
        preferred_days = sorted(day_name_counts.keys(), 
                              key=lambda d: day_name_counts[d], reverse=True)
        
        return {
            "preferred_days": preferred_days,
            "day_distribution": day_name_counts,
            "weekday_vs_weekend": {
                "weekday_events": sum(count for day, count in day_counts.items() if day < 5),
                "weekend_events": sum(count for day, count in day_counts.items() if day >= 5)
            }
        }
    
    def _analyze_seasonal_trends(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze seasonal scheduling trends"""
        
        month_counts = {}
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        for event in events:
            try:
                start_time = datetime.fromisoformat(event["start_datetime"].replace('Z', ''))
                month = start_time.month
                month_counts[month] = month_counts.get(month, 0) + 1
            except:
                continue
        
        # Convert to month names
        month_name_counts = {month_names[month-1]: count for month, count in month_counts.items()}
        
        return {
            "busiest_months": sorted(month_name_counts.keys(), 
                                   key=lambda m: month_name_counts[m], reverse=True)[:3],
            "monthly_distribution": month_name_counts
        }
    
    def _generate_insights(self, patterns: Dict[str, Any], 
                          events: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable insights from patterns"""
        
        insights = []
        
        # Time preference insights
        time_prefs = patterns.get("preferred_hours", {})
        if time_prefs.get("peak_hours"):
            peak_hour = time_prefs["peak_hours"][0]
            insights.append(f"Your most scheduled hour is {peak_hour}:00")
        
        # Success rate insights
        success_patterns = patterns.get("success_patterns", {})
        success_rate = success_patterns.get("overall_success_rate", 0)
        if success_rate < 0.8:
            insights.append("Consider reviewing your scheduling patterns to improve meeting success rates")
        
        # Duration insights
        duration_patterns = patterns.get("duration_patterns", {})
        avg_duration = duration_patterns.get("average_duration", 60)
        if avg_duration > 90:
            insights.append("Your meetings tend to be long - consider breaking them into shorter sessions")
        
        return insights
    
    def _generate_pattern_recommendations(self, patterns: Dict[str, Any], 
                                        insights: List[str]) -> List[str]:
        """Generate recommendations based on patterns and insights"""
        
        recommendations = []
        
        # Time-based recommendations
        time_prefs = patterns.get("preferred_hours", {})
        if time_prefs.get("best_success_hours"):
            best_hour = time_prefs["best_success_hours"][0]
            recommendations.append(f"Schedule important meetings around {best_hour}:00 for better success rates")
        
        # Day-based recommendations
        day_prefs = patterns.get("day_preferences", {})
        weekday_weekend = day_prefs.get("weekday_vs_weekend", {})
        if weekday_weekend.get("weekend_events", 0) > 0:
            recommendations.append("Consider moving weekend meetings to weekdays for better attendance")
        
        # Duration recommendations
        duration_prefs = patterns.get("duration_patterns", {})
        if duration_prefs.get("common_durations"):
            common_duration = duration_prefs["common_durations"][0]
            recommendations.append(f"Your optimal meeting duration appears to be {common_duration} minutes")
        
        return recommendations
    
    def _initialize_embedding_model(self):
        """Initialize sentence transformer model for RAG"""
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.model_type != "rag":
            return False
        
        try:
            # Use a lightweight, fast model for embeddings
            model_name = "all-MiniLM-L6-v2"  # 384-dimensional embeddings
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"Loaded sentence transformer model: {model_name}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load embedding model: {e}")
            return False
    
    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text using sentence transformer"""
        
        if not self.embedding_model:
            if not self._initialize_embedding_model():
                return None
        
        try:
            # Check cache first
            if text in self.embeddings_cache:
                return self.embeddings_cache[text]
            
            # Generate embedding
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            
            # Cache the embedding
            self.embeddings_cache[text] = embedding
            
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _build_knowledge_base(self, training_data: List[Dict[str, Any]]):
        """Build knowledge base from training data for RAG"""
        
        self.knowledge_base = []
        
        for example in training_data:
            # Create contextual text for embedding
            context_text = self._create_context_text(example)
            
            # Generate embedding
            embedding = self._embed_text(context_text)
            
            if embedding is not None:
                self.knowledge_base.append({
                    "text": context_text,
                    "embedding": embedding,
                    "data": example,
                    "success": example.get("success", False),
                    "metadata": {
                        "timestamp": example.get("timestamp", datetime.utcnow().isoformat()),
                        "event_type": example.get("event_type", "meeting")
                    }
                })
        
        self.logger.info(f"Built knowledge base with {len(self.knowledge_base)} examples")
    
    def _create_context_text(self, event_data: Dict[str, Any]) -> str:
        """Create contextual text representation of event for embedding"""
        
        parts = []
        
        # Title and description
        if event_data.get("title"):
            parts.append(f"Event: {event_data['title']}")
        
        if event_data.get("description"):
            parts.append(f"Description: {event_data['description']}")
        
        # Time context
        if event_data.get("hour_of_day"):
            parts.append(f"Time: {event_data['hour_of_day']}:00")
        
        if event_data.get("day_of_week"):
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_name = days[event_data["day_of_week"]] if event_data["day_of_week"] < 7 else "Unknown"
            parts.append(f"Day: {day_name}")
        
        # Duration
        if event_data.get("duration_minutes"):
            parts.append(f"Duration: {event_data['duration_minutes']} minutes")
        
        # Attendees
        if event_data.get("has_attendees"):
            parts.append("Has attendees")
        
        # Work hours
        if event_data.get("is_work_hours"):
            parts.append("During work hours")
        
        # Success context
        if "success" in event_data:
            parts.append(f"Success: {'Yes' if event_data['success'] else 'No'}")
        
        return " | ".join(parts)
    
    def _retrieve_similar_examples(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar examples from knowledge base using semantic similarity"""
        
        if not self.knowledge_base or not self.embedding_model:
            return []
        
        # Generate query embedding
        query_embedding = self._embed_text(query_text)
        if query_embedding is None:
            return []
        
        # Calculate similarities
        similarities = []
        for item in self.knowledge_base:
            try:
                # Cosine similarity
                similarity = np.dot(query_embedding, item["embedding"]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(item["embedding"])
                )
                similarities.append({
                    "item": item,
                    "similarity": similarity
                })
            except Exception as e:
                self.logger.warning(f"Similarity calculation failed: {e}")
                continue
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        self.model_metrics["rag_retrievals"] += 1
        
        return [item["item"] for item in similarities[:k]]
    
    def _rag_predict_success(self, event_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Predict success using RAG approach"""
        
        # Create query text from event data
        query_text = self._create_context_text(event_data)
        
        # Retrieve similar examples
        similar_examples = self._retrieve_similar_examples(query_text, k=10)
        
        if not similar_examples:
            return 0.5, ["No similar examples found"]  # Default probability
        
        # Calculate success probability based on similar examples
        successes = sum(1 for ex in similar_examples if ex["data"].get("success", False))
        probability = successes / len(similar_examples)
        
        # Generate explanations
        explanations = []
        explanations.append(f"Based on {len(similar_examples)} similar events")
        explanations.append(f"{successes} were successful ({probability:.1%} success rate)")
        
        # Add contextual insights
        if probability > 0.8:
            explanations.append("This looks like a high-success event pattern")
        elif probability < 0.4:
            explanations.append("Similar events have had challenges - consider adjustments")
        
        return probability, explanations
    
    async def _validate_suggestion(self, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ML suggestions against schema"""
        
        if not VALIDATION_AVAILABLE:
            return suggestion  # Skip validation if not available
        
        try:
            # Determine suggestion type and validate
            if suggestion.get("type") == "event":
                validation_result = validate_event(suggestion.get("data", {}))
            elif suggestion.get("type") == "task": 
                validation_result = validate_task(suggestion.get("data", {}))
            else:
                return suggestion  # Unknown type, skip validation
            
            if not validation_result.valid:
                self.logger.warning(f"ML suggestion validation failed: {validation_result.errors}")
                # Skip invalid suggestions
                return None
            
            return suggestion
            
        except Exception as e:
            self.logger.error(f"Suggestion validation error: {e}")
            return suggestion  # Return original on error 