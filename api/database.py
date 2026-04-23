"""SQLite database module for prediction logging."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import logger


class PredictionDatabase:
    """SQLite database for logging predictions."""
    
    def __init__(self, db_path: str = "predictions.db"):
        """Initialize database connection and schema.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
    
    def _initialize_schema(self) -> None:
        """Create prediction logs table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        input_features TEXT NOT NULL,
                        prediction REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
            logger.info("Prediction database initialized at %s", self.db_path)
        except sqlite3.Error as e:
            logger.error("Error initializing prediction database: %s", str(e))
            raise
    
    def log_prediction(
        self,
        model_version: str,
        model_name: str,
        task_type: str,
        input_features: Dict[str, Any],
        prediction: float | int,
        confidence_score: float,
        timestamp: str,
    ) -> int:
        """Log a prediction to the database.
        
        Args:
            model_version: Version identifier of the model
            model_name: Name of the model
            task_type: Type of task (classification/regression)
            input_features: Input feature dictionary
            prediction: Prediction value
            confidence_score: Model confidence score
            timestamp: ISO format timestamp of prediction
            
        Returns:
            Database row ID
        """
        try:
            features_json = json.dumps(input_features)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO predictions 
                    (timestamp, model_version, model_name, task_type, input_features, prediction, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (timestamp, model_version, model_name, task_type, features_json, prediction, confidence_score)
                )
                conn.commit()
                row_id = cursor.lastrowid
            
            logger.info("Prediction logged with ID: %s", row_id)
            return row_id
        except sqlite3.Error as e:
            logger.error("Error logging prediction: %s", str(e))
            raise
    
    def get_recent_predictions(self, limit: int = 100) -> list[Dict[str, Any]]:
        """Retrieve recent predictions from database.
        
        Args:
            limit: Maximum number of predictions to retrieve
            
        Returns:
            List of prediction records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM predictions 
                    ORDER BY created_at DESC 
                    LIMIT ?
                    """,
                    (limit,)
                )
                rows = cursor.fetchall()
                
            predictions = []
            for row in rows:
                pred = dict(row)
                pred["input_features"] = json.loads(pred["input_features"])
                predictions.append(pred)
            
            return predictions
        except sqlite3.Error as e:
            logger.error("Error retrieving predictions: %s", str(e))
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics from database.
        
        Returns:
            Dictionary with statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT 
                        COUNT(*) as total_predictions,
                        COUNT(DISTINCT model_version) as unique_models,
                        AVG(confidence_score) as avg_confidence,
                        MIN(confidence_score) as min_confidence,
                        MAX(confidence_score) as max_confidence
                    FROM predictions
                    """
                )
                stats = dict(cursor.fetchone())
            
            return stats
        except sqlite3.Error as e:
            logger.error("Error retrieving statistics: %s", str(e))
            raise


# Global database instance
_db_instance: Optional[PredictionDatabase] = None


def get_prediction_db(db_path: str = "predictions.db") -> PredictionDatabase:
    """Get or create global prediction database instance.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        PredictionDatabase instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = PredictionDatabase(db_path)
    return _db_instance
