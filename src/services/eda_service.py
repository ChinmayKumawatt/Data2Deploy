"""
EDA (Exploratory Data Analysis) Service for Data2Deploy.
Provides data insights, analysis, and visualization recommendations.
"""

import json
import io
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import scipy.stats as stats


class EDAService:
    """Service for exploratory data analysis and insights."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA service with a DataFrame.
        
        Args:
            df: Input DataFrame to analyze
        """
        self.df = df
        self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self._all_cols = df.columns.tolist()
    
    # ==================== Data Overview ====================
    
    def get_data_overview(self) -> Dict[str, Any]:
        """Get comprehensive data overview."""
        return {
            "shape": {
                "rows": int(self.df.shape[0]),
                "columns": int(self.df.shape[1])
            },
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": self._get_missing_values(),
            "memory_usage_mb": float(self.df.memory_usage(deep=True).sum() / 1024 / 1024),
            "duplicates": int(self.df.duplicated().sum()),
            "duplicate_pct": float((self.df.duplicated().sum() / len(self.df)) * 100)
        }
    
    def _get_missing_values(self) -> Dict[str, Any]:
        """Get missing value statistics."""
        missing = self.df.isnull().sum()
        return {
            col: {
                "count": int(missing[col]),
                "percentage": float((missing[col] / len(self.df)) * 100)
            }
            for col in self.df.columns if missing[col] > 0
        }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for numeric columns."""
        stats_dict = {}
        
        for col in self._numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                stats_dict[col] = {
                    "count": int(len(data)),
                    "mean": float(data.mean()),
                    "median": float(data.median()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "q25": float(data.quantile(0.25)),
                    "q75": float(data.quantile(0.75)),
                    "max": float(data.max()),
                    "skewness": float(stats.skew(data)),
                    "kurtosis": float(stats.kurtosis(data))
                }
        
        return stats_dict
    
    def get_categorical_summary(self) -> Dict[str, Any]:
        """Get summary for categorical columns."""
        cat_summary = {}
        
        for col in self._categorical_cols:
            value_counts = self.df[col].value_counts()
            cat_summary[col] = {
                "unique_count": int(self.df[col].nunique()),
                "top_values": {
                    str(k): int(v) for k, v in value_counts.head(10).items()
                },
                "top_pct": {
                    str(k): float((v / len(self.df)) * 100) 
                    for k, v in value_counts.head(10).items()
                }
            }
        
        return cat_summary
    
    def get_correlation_matrix(self) -> Dict[str, Any]:
        """Get correlation matrix for numeric columns."""
        if len(self._numeric_cols) < 2:
            return {"message": "Not enough numeric columns for correlation"}
        
        corr_matrix = self.df[self._numeric_cols].corr()
        
        # Convert to JSON-serializable format
        return {
            "columns": self._numeric_cols,
            "data": corr_matrix.values.tolist(),
            "correlations": corr_matrix.to_dict()
        }
    
    # ==================== Insight Detection ====================
    
    def detect_outliers(self) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        outliers = {}
        
        for col in self._numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = len(data[(data < lower_bound) | (data > upper_bound)])
                outlier_pct = (outlier_count / len(data)) * 100
                
                if outlier_count > 0:
                    outliers[col] = {
                        "count": int(outlier_count),
                        "percentage": float(outlier_pct),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound)
                    }
        
        return outliers
    
    def detect_skewness(self, threshold: float = 0.5) -> Dict[str, Any]:
        """Detect skewed distributions."""
        skewed_cols = {}
        
        for col in self._numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 2:
                skewness = stats.skew(data)
                
                if abs(skewness) > threshold:
                    skewed_cols[col] = {
                        "skewness": float(skewness),
                        "direction": "right" if skewness > 0 else "left",
                        "severity": "moderate" if abs(skewness) < 1 else "high"
                    }
        
        return skewed_cols
    
    def detect_strong_correlations(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Detect strong correlations between features."""
        if len(self._numeric_cols) < 2:
            return []
        
        corr_matrix = self.df[self._numeric_cols].corr()
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    strong_corrs.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": float(corr_value),
                        "relationship": "positive" if corr_value > 0 else "negative"
                    })
        
        return sorted(strong_corrs, key=lambda x: abs(x["correlation"]), reverse=True)
    
    def detect_constant_features(self) -> Dict[str, Any]:
        """Detect features with constant or near-constant values."""
        constant_features = {}
        
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            unique_pct = (unique_count / len(self.df)) * 100
            
            if unique_count <= 1:
                constant_features[col] = {
                    "type": "constant",
                    "unique_values": int(unique_count)
                }
            elif unique_pct < 1:  # Less than 1% unique values
                constant_features[col] = {
                    "type": "near_constant",
                    "unique_values": int(unique_count),
                    "unique_percentage": float(unique_pct)
                }
        
        return constant_features
    
    # ==================== Insights & Recommendations ====================
    
    def get_all_insights(self) -> Dict[str, Any]:
        """Get all detected insights."""
        return {
            "outliers": self.detect_outliers(),
            "skewness": self.detect_skewness(),
            "strong_correlations": self.detect_strong_correlations(),
            "constant_features": self.detect_constant_features()
        }
    
    def get_feature_engineering_recommendations(self) -> List[Dict[str, Any]]:
        """Get feature engineering recommendations based on insights."""
        recommendations = []
        
        # Skewness recommendations
        skewed = self.detect_skewness()
        for col, info in skewed.items():
            recommendations.append({
                "feature": col,
                "issue": f"Skewed distribution ({info['direction']} skew: {info['skewness']:.2f})",
                "recommendation": "Log transform" if info['direction'] == 'right' else "Box-Cox transform",
                "priority": "high" if info['severity'] == 'high' else "medium",
                "reason": "Can improve model performance by normalizing distribution"
            })
        
        # Outlier recommendations
        outliers = self.detect_outliers()
        for col, info in outliers.items():
            if info['percentage'] > 5:
                recommendations.append({
                    "feature": col,
                    "issue": f"Outliers detected ({info['percentage']:.1f}%)",
                    "recommendation": "Consider robust scaling or outlier removal",
                    "priority": "medium",
                    "reason": "Outliers can negatively impact model training"
                })
        
        # Constant feature recommendations
        constant = self.detect_constant_features()
        for col in constant:
            recommendations.append({
                "feature": col,
                "issue": "Constant or near-constant feature",
                "recommendation": "Remove this feature",
                "priority": "high",
                "reason": "Non-informative features don't contribute to predictions"
            })
        
        # Correlation recommendations
        correlations = self.detect_strong_correlations(threshold=0.85)
        for corr in correlations[:5]:  # Top 5
            recommendations.append({
                "feature": f"{corr['feature_1']} & {corr['feature_2']}",
                "issue": f"High correlation ({corr['correlation']:.2f})",
                "recommendation": "Consider removing one or using dimensionality reduction",
                "priority": "medium",
                "reason": "Multicollinearity can reduce model interpretability"
            })
        
        return sorted(recommendations, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]])
    
    # ==================== Plot Suggestions ====================
    
    def suggest_plots(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get smart plot suggestions based on data types."""
        suggestions = {
            "univariate": [],
            "bivariate": [],
            "multivariate": []
        }
        
        # Univariate plots
        for col in self._numeric_cols[:10]:  # Limit to first 10
            suggestions["univariate"].append({
                "name": "Histogram",
                "plot_type": "histogram",
                "x": col,
                "reason": "Visualize distribution"
            })
            suggestions["univariate"].append({
                "name": "Box Plot",
                "plot_type": "boxplot",
                "x": col,
                "reason": "Identify outliers and quartiles"
            })
        
        for col in self._categorical_cols[:10]:
            suggestions["univariate"].append({
                "name": "Bar Chart",
                "plot_type": "bar",
                "x": col,
                "reason": "Show value frequencies"
            })
        
        # Bivariate plots
        if len(self._numeric_cols) >= 2:
            for i, col1 in enumerate(self._numeric_cols[:5]):
                for col2 in self._numeric_cols[i+1:6]:
                    suggestions["bivariate"].append({
                        "name": f"Scatter: {col1} vs {col2}",
                        "plot_type": "scatter",
                        "x": col1,
                        "y": col2,
                        "reason": "Explore relationship"
                    })
        
        # Correlation heatmap
        if len(self._numeric_cols) >= 3:
            suggestions["multivariate"].append({
                "name": "Correlation Heatmap",
                "plot_type": "heatmap",
                "reason": "Visualize all correlations"
            })
        
        return suggestions
    
    def get_plot_config(self, plot_type: str, x: str, y: str = None, group: str = None) -> Dict[str, Any]:
        """Get configuration for generating a specific plot."""
        config = {
            "plot_type": plot_type,
            "x": x,
            "y": y,
            "group": group,
            "data_summary": {}
        }
        
        # Get relevant data summary
        if x in self._numeric_cols:
            config["data_summary"]["x_type"] = "numeric"
            config["data_summary"]["x_stats"] = {
                "min": float(self.df[x].min()),
                "max": float(self.df[x].max()),
                "mean": float(self.df[x].mean())
            }
        else:
            config["data_summary"]["x_type"] = "categorical"
            config["data_summary"]["x_values"] = self.df[x].unique().tolist()[:20]
        
        if y and y in self._numeric_cols:
            config["data_summary"]["y_type"] = "numeric"
            config["data_summary"]["y_stats"] = {
                "min": float(self.df[y].min()),
                "max": float(self.df[y].max()),
                "mean": float(self.df[y].mean())
            }
        elif y:
            config["data_summary"]["y_type"] = "categorical"
        
        return config
    
    # ==================== Data Preview ====================
    
    def get_data_preview(self, n_rows: int = 100) -> Dict[str, Any]:
        """Get preview of the data."""
        preview_df = self.df.head(n_rows)
        
        return {
            "columns": self.df.columns.tolist(),
            "data": preview_df.values.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in preview_df.dtypes.items()},
            "rows_shown": len(preview_df),
            "total_rows": len(self.df)
        }


def create_eda_from_csv(file_path: str) -> EDAService:
    """Create EDA service from CSV file."""
    df = pd.read_csv(file_path)
    return EDAService(df)


def create_eda_from_bytes(file_bytes: bytes) -> EDAService:
    """Create EDA service from CSV bytes."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    return EDAService(df)


def create_eda_from_dataframe(df: pd.DataFrame) -> EDAService:
    """Create EDA service from DataFrame."""
    return EDAService(df)
