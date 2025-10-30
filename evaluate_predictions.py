"""
Evaluation Pipeline for LLM Predictions
Scores LLM outputs against ground truth targets.
"""

import json
from typing import List, Dict, Tuple
from data_pipeline import DataPipeline


class PredictionEvaluator:
    """
    Evaluator for comparing LLM predictions against ground truth.
    """
    
    def __init__(self):
        self.results = []
        
    def parse_llm_response(self, response: str) -> str:
        """
        Parse and clean the LLM response to extract the JSON string.
        Handles various formats the LLM might output.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Cleaned JSON string
        """
        response = response.strip()
        
        # Try to extract JSON if wrapped in markdown code blocks
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            response = response[start:end].strip()
        elif '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            response = response[start:end].strip()
        
        # Remove any leading/trailing text before/after the JSON
        # Find the first { and last }
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace != -1 and last_brace != -1:
            response = response[first_brace:last_brace+1]
        
        return response
    
    def prepare_llm_responses(self, raw_responses: List[str]) -> List[str]:
        """
        Prepare array of LLM response strings.
        
        Args:
            raw_responses: List of raw LLM output strings
            
        Returns:
            List of cleaned response strings
        """
        return [self.parse_llm_response(resp) for resp in raw_responses]
    
    def exact_match_score(self, prediction: str, target: str) -> int:
        """
        Compare prediction and target strings for exact match.
        
        Args:
            prediction: LLM prediction string
            target: Ground truth target string
            
        Returns:
            1 if exact match, 0 otherwise
        """
        # Normalize both strings (strip whitespace)
        pred_normalized = prediction.strip()
        target_normalized = target.strip()
        
        return 1 if pred_normalized == target_normalized else 0
    
    def evaluate(self, predictions: List[str], targets: List[str]) -> Dict:
        """
        Evaluate predictions against targets using exact string matching.
        
        Args:
            predictions: List of LLM prediction strings
            targets: List of ground truth target strings
            
        Returns:
            Dictionary with evaluation results
        """
        if len(predictions) != len(targets):
            raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(targets)} targets")
        
        total_trials = len(targets)
        correct_count = 0
        self.results = []
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            score = self.exact_match_score(pred, target)
            correct_count += score
            
            self.results.append({
                'trial_id': i,
                'prediction': pred,
                'target': target,
                'score': score,
                'match': score == 1
            })
        
        accuracy = correct_count / total_trials if total_trials > 0 else 0.0
        
        return {
            'total_trials': total_trials,
            'correct_count': correct_count,
            'incorrect_count': total_trials - correct_count,
            'accuracy': accuracy,
            'accuracy_percentage': accuracy * 100
        }
    
    def get_detailed_results(self) -> List[Dict]:
        """
        Get detailed per-trial results.
        
        Returns:
            List of dictionaries with trial-by-trial results
        """
        return self.results
    
    def print_report(self, metrics: Dict, show_details: bool = True):
        """
        Print a formatted evaluation report.
        
        Args:
            metrics: Dictionary returned from evaluate()
            show_details: Whether to show trial-by-trial details
        """
        print("\n" + "=" * 80)
        print(" EVALUATION REPORT")
        print("=" * 80)
        
        print(f"\nTotal Trials: {metrics['total_trials']}")
        print(f"Correct: {metrics['correct_count']}")
        print(f"Incorrect: {metrics['incorrect_count']}")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy_percentage']:.2f}%)")
        
        if show_details and self.results:
            print("\n" + "-" * 80)
            print(" TRIAL-BY-TRIAL RESULTS")
            print("-" * 80)
            
            for result in self.results:
                status = "✓ CORRECT" if result['match'] else "✗ INCORRECT"
                print(f"\nTrial {result['trial_id']}: {status} (Score: {result['score']}/1)")
                print(f"  Target:     {result['target']}")
                print(f"  Prediction: {result['prediction']}")
        
        print("\n" + "=" * 80)
    
    def save_results(self, metrics: Dict, output_file: str):
        """
        Save evaluation results to a JSON file.
        
        Args:
            metrics: Dictionary returned from evaluate()
            output_file: Path to output file
        """
        output = {
            'summary': metrics,
            'detailed_results': self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
