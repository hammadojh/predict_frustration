#!/usr/bin/env python3
"""
Test Frustration Detection Model

This script loads a trained frustration detection model and runs predictions
on example texts or custom inputs.

Usage:
    python3 scripts/test_frustration_detection.py
    python3 scripts/test_frustration_detection.py --text "Your custom text here"
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from scripts
sys.path.append(str(Path(__file__).parent.parent))

from scripts.frustration_detection import FrustrationDetector
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(checkpoint_path="./frustration_detection_results/checkpoint-19172"):
    """
    Load a trained frustration detection model.
    
    Args:
        checkpoint_path (str): Path to the saved model checkpoint
        
    Returns:
        FrustrationDetector: Loaded detector with trained model
    """
    if not os.path.exists(checkpoint_path):
        # Try alternative path
        checkpoint_path = "./results"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No trained model found. Please run training first with: "
                f"python3 scripts/frustration_detection.py"
            )
    
    logger.info(f"Loading trained model from: {checkpoint_path}")
    
    # Create detector and load components
    detector = FrustrationDetector()
    detector.tokenizer = RobertaTokenizerFast.from_pretrained("SamLowe/roberta-base-go_emotions")
    detector.model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)
    
    logger.info("✅ Model loaded successfully!")
    return detector

def run_example_tests(detector):
    """Run predictions on example texts."""
    
    test_cases = [
        # Frustrated examples
        "I'm so annoyed that the system keeps giving me wrong answers!",
        "This is frustrating, I've been trying to solve this for hours.",
        "I am really angry about this situation.",
        "I hate when things don't work properly!",
        "This is driving me crazy, nothing is working!",
        
        # Not frustrated examples  
        "Thank you so much for your help, this is exactly what I needed.",
        "Great! Everything is working perfectly now.",
        "The weather is nice today.",
        "I'm having a wonderful time with my friends.",
        "This solution works really well."
    ]
    
    print('\n📊 Testing Frustration Detection Model')
    print('=' * 60)
    
    for text in test_cases:
        try:
            result = detector.predict(text)
            label = result['label']
            confidence = result['confidence']
            emoji = '😤' if label == 'frustrated' else '😊'
            
            print(f'{emoji} {label.upper():>13} | {confidence:6.1%} | {text}')
            
        except Exception as e:
            print(f'❌ ERROR | Failed to predict: {text}')
            print(f'   Error: {str(e)}')
    
    print('=' * 60)

def interactive_mode(detector):
    """Run interactive mode for custom text input."""
    
    print('\n🎯 Interactive Frustration Detection')
    print('Enter text to analyze (or "quit" to exit):')
    print('-' * 50)
    
    while True:
        try:
            text = input('\n> ').strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print('Goodbye!')
                break
                
            if not text:
                continue
                
            result = detector.predict(text)
            label = result['label']
            confidence = result['confidence']
            emoji = '😤' if label == 'frustrated' else '😊'
            
            print(f'{emoji} {label.upper()} (confidence: {confidence:.1%})')
            
            # Show detailed probabilities
            probs = result['probabilities']
            print(f'   Not Frustrated: {probs["not_frustrated"]:.1%}')
            print(f'   Frustrated: {probs["frustrated"]:.1%}')
            
        except KeyboardInterrupt:
            print('\nGoodbye!')
            break
        except Exception as e:
            print(f'❌ Error: {str(e)}')

def main():
    """Main function to run frustration detection tests."""
    
    parser = argparse.ArgumentParser(description='Test frustration detection model')
    parser.add_argument('--text', type=str, help='Single text to analyze')
    parser.add_argument('--checkpoint', type=str, 
                       default='./frustration_detection_results/checkpoint-19172',
                       help='Path to model checkpoint')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--examples-only', action='store_true',
                       help='Run only example tests, skip interactive mode')
    
    args = parser.parse_args()
    
    try:
        # Load the trained model
        detector = load_trained_model(args.checkpoint)
        
        # Single text prediction
        if args.text:
            result = detector.predict(args.text)
            label = result['label']
            confidence = result['confidence']
            emoji = '😤' if label == 'frustrated' else '😊'
            
            print(f'\n📝 Text: "{args.text}"')
            print(f'{emoji} Prediction: {label.upper()} (confidence: {confidence:.1%})')
            
            probs = result['probabilities']
            print(f'\nDetailed probabilities:')
            print(f'  Not Frustrated: {probs["not_frustrated"]:.1%}')
            print(f'  Frustrated: {probs["frustrated"]:.1%}')
            return
        
        # Run example tests
        run_example_tests(detector)
        
        # Interactive mode (unless disabled)
        if not args.examples_only:
            if args.interactive or input('\nRun interactive mode? (y/n): ').lower().startswith('y'):
                interactive_mode(detector)
        
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()