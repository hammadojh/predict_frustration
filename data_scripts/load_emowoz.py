from datasets import load_dataset
import pandas as pd
import json

def load_emowoz_dataset():
    """Load and explore the EmoWOZ dataset"""
    print("Loading EmoWOZ dataset...")
    
    # Load the dataset
    emo = load_dataset("hhu-dsml/emowoz")
    print("Dataset loaded successfully")
    print(f"Train: {len(emo['train'])}")
    print(f"Validation: {len(emo['validation'])}")
    print(f"Test: {len(emo['test'])}")
    
    # Explore structure
    print("\nSample entry:", emo['train'][0])
    
    # Print some basic statistics
    print("\nDataset structure:")
    for split_name, split_data in emo.items():
        print(f"{split_name}: {len(split_data)} dialogues")
        
        # Analyze emotions in the first few dialogues - use proper dataset selection
        emotions = []
        sample_data = split_data.select(range(min(100, len(split_data))))  # First 100 dialogues
        
        for dialogue in sample_data:
            log = dialogue['log']
            emotion_list = log['emotion']
            text_list = log['text']
            
            # Process user turns (even indices: 0, 2, 4, ...)
            for i in range(0, len(emotion_list), 2):  # User turns
                emotion = emotion_list[i]
                if emotion != -1:  # -1 means no emotion label
                    # Map emotion numbers to labels
                    emotion_map = {
                        0: 'neutral',
                        1: 'fearful', 
                        2: 'dissatisfied',
                        3: 'apologetic',
                        4: 'abusive',
                        5: 'excited',
                        6: 'satisfied'
                    }
                    emotion_label = emotion_map.get(emotion, f'unknown_{emotion}')
                    emotions.append(emotion_label)
        
        emotion_counts = pd.Series(emotions).value_counts()
        print(f"Emotion distribution in {split_name} (first 100 dialogues):")
        print(emotion_counts)
        print(f"Total user turns with emotions: {len(emotions)}")
        print()
    
    return emo

if __name__ == "__main__":
    dataset = load_emowoz_dataset() 