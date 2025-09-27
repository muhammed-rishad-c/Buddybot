from src.exception.custom_exception import CustomException
from src.loggings.custom_loggings import logging
import os,sys
from transformers import AutoTokenizer
import numpy as np
from src.config.config_entity import DataTransformationConfig
from src.config.artifact_entity import DataIngestionArtifact,DataTransformationArtifact
from src.utils.main_utils import read_df,save_np_array
from tqdm import tqdm
from src.config.config_entity import TrainingPipelineConfig
from datasets import Dataset
from ast import literal_eval
import re
import json

class DataTransformation:
    def __init__(self,data_transformation_config=DataTransformationConfig,
                 data_ingestion_artifact=DataIngestionArtifact):
        self.data_transformation_config=data_transformation_config
        self.data_ingestion_artifact=data_ingestion_artifact
        self.tokenizer=AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token=self.tokenizer.eos_token
    
    def debug_single_dialog(self, dialog_sample):
        """Debug function to test dialog processing on a single sample"""
        try:
            print(f"Original dialog: {dialog_sample}")
            print(f"Dialog type: {type(dialog_sample)}")
            
            formatted = self.format_dialog(dialog_sample)
            print(f"Formatted dialog: {formatted}")
            
            if formatted:
                pairs = self.create_training_pairs(formatted)
                print(f"Training pairs created: {len(pairs)}")
                for i, pair in enumerate(pairs[:3]):  # Show first 3 pairs
                    print(f"Pair {i+1}: {pair['text'][:100]}...")
            else:
                print("No formatted dialog produced")
                
        except Exception as e:
            print(f"Error in debug: {e}")
            import traceback
            traceback.print_exc()
    
    def split_concatenated_dialog(self, dialog_text):
        """
        Try to split a concatenated dialog back into individual turns
        This is a heuristic approach since the original structure is lost
        """
        try:
            # Remove extra spaces and clean up
            dialog_text = re.sub(r'\s+', ' ', dialog_text.strip())
            
            # Split by sentence endings followed by capital letters (common dialog pattern)
            # This is imperfect but works for many cases
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', dialog_text)
            
            # Filter out very short sentences (likely artifacts)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
            
            # If we didn't get good splits, try splitting by common dialog patterns
            if len(sentences) < 3:
                # Try splitting on common question/response patterns
                potential_splits = re.split(r'(?<=[.!?])\s+(?=(?:Yes|No|Well|Oh|I|You|Can|Do|What|How|Why|Where|When))', dialog_text)
                if len(potential_splits) > len(sentences):
                    sentences = [s.strip() for s in potential_splits if len(s.strip()) > 3]
            
            # If still no good splits, split roughly in half or thirds
            if len(sentences) < 2:
                mid = len(dialog_text) // 2
                sentences = [dialog_text[:mid].strip(), dialog_text[mid:].strip()]
            
            return sentences
            
        except Exception as e:
            logging.error(f"Error splitting dialog: {e}")
            return [dialog_text]  # Return as single turn if splitting fails
    
    def format_dialog(self, dialog_input):
        """
        Format dialog from various input types
        Handles the concatenated format from your current CSV
        """
        try:
            dialog_list = None
            
            # Handle different input types
            if isinstance(dialog_input, str):
                dialog_input = dialog_input.strip()
                
                # Case 1: JSON string (if using the fixed ingestion)
                if dialog_input.startswith('[') and dialog_input.endswith(']'):
                    try:
                        # First try JSON parsing
                        dialog_list = json.loads(dialog_input)
                    except json.JSONDecodeError:
                        try:
                            # Fall back to literal_eval
                            dialog_list = literal_eval(dialog_input)
                        except (ValueError, SyntaxError):
                            logging.warning(f"Could not parse dialog string: {dialog_input[:100]}...")
                            return ""
                else:
                    # Case 2: Plain text
                    return ""
                    
            elif isinstance(dialog_input, list):
                dialog_list = dialog_input
            else:
                logging.warning(f"Unexpected input type: {type(dialog_input)}")
                return ""
            
            if not isinstance(dialog_list, list) or len(dialog_list) == 0:
                return ""
            
            # Check if this is the concatenated format (single string in list)
            if len(dialog_list) == 1 and isinstance(dialog_list[0], str):
                # This is the concatenated format - try to split it
                logging.debug("Detected concatenated dialog format, attempting to split...")
                dialog_turns = self.split_concatenated_dialog(dialog_list[0])
                
                # If we couldn't split well, create a simple user-bot alternating pattern
                if len(dialog_turns) < 2:
                    # Just use the whole text as alternating user-bot
                    text = dialog_list[0].strip()
                    if len(text) < 10:  # Too short to be useful
                        return ""
                    
                    # Split roughly in half for user and bot
                    mid = len(text) // 2
                    # Find a good split point near the middle
                    split_point = mid
                    for i in range(max(0, mid-50), min(len(text), mid+50)):
                        if text[i] in '.!?':
                            split_point = i + 1
                            break
                    
                    user_part = text[:split_point].strip()
                    bot_part = text[split_point:].strip()
                    
                    if user_part and bot_part:
                        return f"[USER] : {user_part} [BOT] : {bot_part}"
                    else:
                        # Fallback: use whole text as user input, generate simple response
                        return f"[USER] : {text} [BOT] : I understand."
                
                # Use the split turns
                dialog_list = dialog_turns
            
            # Now format the turns (assuming alternating user/bot)
            formatted_turns = []
            valid_turns = 0
            
            for i, turn in enumerate(dialog_list):
                if not isinstance(turn, str):
                    continue
                    
                cleaned_turn = turn.strip().replace('\n', ' ').replace('\r', ' ')
                if not cleaned_turn or len(cleaned_turn) < 3:
                    continue
                
                # Alternate between USER and BOT
                if i % 2 == 0:
                    formatted_turns.append(f"[USER] : {cleaned_turn}")
                else:
                    formatted_turns.append(f"[BOT] : {cleaned_turn}")
                
                valid_turns += 1
            
            if valid_turns == 0:
                return ""
            
            # If we only have user turns, add a simple bot response
            if valid_turns == 1 and formatted_turns[0].startswith("[USER]"):
                formatted_turns.append("[BOT] : I understand.")
                valid_turns = 2
            
            result = " ".join(formatted_turns)
            return result
            
        except Exception as e:
            logging.error(f"Error formatting dialog: {e}")
            return ""
        
        
    def create_training_pairs(self, formatted_dialogue, context_size=2):
        """
        Create training pairs from formatted dialogue
        """
        try:
            if not formatted_dialogue or formatted_dialogue.strip() == "":
                return []

            # Use regex to extract USER and BOT turns
            import re
            
            pattern = r'\[(USER|BOT)\]\s*:\s*([^[]+?)(?=\s*\[(?:USER|BOT)\]|$)'
            matches = re.findall(pattern, formatted_dialogue, re.DOTALL)
            
            if not matches:
                logging.debug(f"No USER/BOT patterns found in: {formatted_dialogue[:150]}...")
                return []
            
            # Build turns and find bot responses
            turns = []
            bot_turn_indices = []
            
            for i, (speaker, content) in enumerate(matches):
                content = content.strip()
                if content:
                    turn = f"[{speaker}] : {content}"
                    turns.append(turn)
                    if speaker == "BOT":
                        bot_turn_indices.append(len(turns) - 1)
            
            logging.debug(f"Found {len(turns)} turns, {len(bot_turn_indices)} bot turns")
            
            if len(bot_turn_indices) == 0:
                logging.debug(f"No bot responses found. All turns: {[t[:50] for t in turns[:3]]}")
                return []
            
            training_pairs = []

            # Create training pairs for each bot response
            for bot_idx in bot_turn_indices:
                # Get context (previous turns)
                start_idx = max(0, bot_idx - context_size)
                
                if start_idx >= bot_idx:
                    continue
                    
                input_context = " ".join(turns[start_idx:bot_idx])
                
                if not input_context.strip():
                    continue
                
                target_response = turns[bot_idx]
                
                # Combine context with target for language modeling
                combined_text = f"{input_context} {target_response}"
                training_pairs.append({"text": combined_text})

            logging.debug(f"Created {len(training_pairs)} training pairs")
            return training_pairs
        
        except Exception as e:
            logging.error(f"Error creating training pairs: {e}")
            logging.error(f"Input was: {formatted_dialogue[:200] if formatted_dialogue else 'None'}...")
            return []
        
        
        
    def initiate_data_transformation(self):
        try:
            logging.info("Data transformation initiated...")
            
            # Load the dataframes
            train_df = read_df(self.data_ingestion_artifact.train_filepath)
            test_df = read_df(self.data_ingestion_artifact.test_filepath)
            logging.info("Data ingestion artifacts loaded")
            
            if 'dialog' not in train_df.columns:
                raise ValueError(f"'dialog' column not found. Available columns: {train_df.columns.tolist()}")
            
            sample_dialog = train_df['dialog'].iloc[0]

            logging.info("Testing format_dialog on samples...")
            for i in range(min(3, len(train_df))):
                test_sample = train_df['dialog'].iloc[i]
                test_formatted = self.format_dialog(test_sample)
                logging.info(f"Sample {i+1} formatted (first 200 chars): {test_formatted[:200]}...")
                
                if test_formatted:
                    test_pairs = self.create_training_pairs(test_formatted)
                    logging.info(f"Sample {i+1} created {len(test_pairs)} training pairs")
                else:
                    logging.warning(f"Sample {i+1} produced no formatted output")
            
            # Format all dialogues
            logging.info("Formatting all dialogues...")
            train_df['formatted_dialog'] = train_df['dialog'].apply(self.format_dialog)
            test_df['formatted_dialog'] = test_df['dialog'].apply(self.format_dialog)
            
            # Check for empty formatted dialogs
            train_empty_mask = train_df['formatted_dialog'].str.len() == 0
            test_empty_mask = test_df['formatted_dialog'].str.len() == 0
            
            empty_train_count = train_empty_mask.sum()
            empty_test_count = test_empty_mask.sum()
            
            logging.info(f"Empty formatted dialogs - Train: {empty_train_count}/{len(train_df)}, Test: {empty_test_count}/{len(test_df)}")
            
            # Remove empty dialogues
            train_df = train_df[~train_empty_mask].reset_index(drop=True)
            test_df = test_df[~test_empty_mask].reset_index(drop=True)
            
            logging.info(f"After filtering - Train: {len(train_df)}, Test: {len(test_df)}")
            
            if len(train_df) == 0:
                raise ValueError("All training dialogues were filtered out")
            if len(test_df) == 0:
                raise ValueError("All test dialogues were filtered out")
            
            # Create training pairs
            logging.info("Creating training pairs...")
            
            train_pairs = []
            successful_train = 0
            failed_train = 0
            
            for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing train data"):
                pairs = self.create_training_pairs(row['formatted_dialog'])
                if pairs:
                    train_pairs.extend(pairs)
                    successful_train += 1
                else:
                    failed_train += 1
                    if failed_train <= 3:  # Log first few failures
                        logging.warning(f"Train dialog {idx} produced no pairs: {row['formatted_dialog'][:100]}...")
                
            test_pairs = []
            successful_test = 0
            failed_test = 0
            
            for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test data"):
                pairs = self.create_training_pairs(row['formatted_dialog'])
                if pairs:
                    test_pairs.extend(pairs)
                    successful_test += 1
                else:
                    failed_test += 1
                    if failed_test <= 3:  # Log first few failures
                        logging.warning(f"Test dialog {idx} produced no pairs: {row['formatted_dialog'][:100]}...")
                
            logging.info(f"Train processing - Success: {successful_train}, Failed: {failed_train}")
            logging.info(f"Test processing - Success: {successful_test}, Failed: {failed_test}")
            logging.info(f"Training pairs created - Train: {len(train_pairs)}, Test: {len(test_pairs)}")
            
            # Validate we have training pairs
            if len(train_pairs) == 0:
                raise ValueError("No training pairs created from train data")
            
            # Handle test data more gracefully
            if len(test_pairs) == 0:
                logging.warning("No training pairs created from test data - this may indicate an issue with test data processing")
                
                # Debug: Show some test samples
                logging.info("Debugging test data:")
                for i in range(min(5, len(test_df))):
                    sample_dialog = test_df['dialog'].iloc[i]
                    formatted = test_df['formatted_dialog'].iloc[i]
                    logging.info(f"Test sample {i}:")
                    logging.info(f"  Raw: {str(sample_dialog)[:150]}...")
                    logging.info(f"  Formatted: {formatted[:150]}...")
                    
                    if formatted:
                        test_pairs_sample = self.create_training_pairs(formatted)
                        logging.info(f"  Pairs created: {len(test_pairs_sample)}")
                        if test_pairs_sample:
                            logging.info(f"  Sample pair: {test_pairs_sample[0]['text'][:100]}...")
                
                # Create minimal test data from train data if needed
                logging.warning("Creating minimal test data from a subset of train data")
                test_subset_size = min(100, len(train_pairs) // 10)  # 10% of train or max 100
                test_pairs = train_pairs[:test_subset_size]
                logging.info(f"Created {len(test_pairs)} test pairs from train data subset")
            
            # Show sample training pairs
            logging.info("Sample training pairs:")
            for i, pair in enumerate(train_pairs[:3]):
                logging.info(f"Pair {i+1}: {pair['text'][:150]}...")
            
            # Create datasets
            logging.info("Creating datasets...")
            train_dataset = Dataset.from_dict({"text": [pair['text'] for pair in train_pairs]})
            test_dataset = Dataset.from_dict({"text": [pair['text'] for pair in test_pairs]})
            
            # Tokenization function
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors=None
                )
            
            # Tokenize datasets
            logging.info("Tokenizing datasets...")
            tokenized_train_dataset = train_dataset.map(
                tokenize_function, 
                batched=True, 
                remove_columns=['text']
            )
            
            tokenized_test_dataset = test_dataset.map(
                tokenize_function, 
                batched=True, 
                remove_columns=['text']
            )
            
            # Debug: Check tokenized columns
            logging.info(f"Tokenized train columns: {tokenized_train_dataset.column_names}")
            logging.info(f"Tokenized test columns: {tokenized_test_dataset.column_names}")
            
            # Verify required columns exist
            required_columns = ['input_ids', 'attention_mask']
            for col in required_columns:
                if col not in tokenized_train_dataset.column_names:
                    raise ValueError(f"Column '{col}' missing from train dataset. Available: {tokenized_train_dataset.column_names}")
                if col not in tokenized_test_dataset.column_names:
                    raise ValueError(f"Column '{col}' missing from test dataset. Available: {tokenized_test_dataset.column_names}")
            
            # Convert to numpy arrays
            logging.info("Converting to numpy arrays...")
            train_input_ids = np.array(tokenized_train_dataset['input_ids'])
            train_attention_mask = np.array(tokenized_train_dataset['attention_mask'])
            
            # Only process test data if we have test pairs
            if len(test_pairs) > 0:
                test_input_ids = np.array(tokenized_test_dataset['input_ids'])
                test_attention_mask = np.array(tokenized_test_dataset['attention_mask'])
                logging.info(f"Arrays created - Train: {train_input_ids.shape}, Test: {test_input_ids.shape}")
            else:
                # Create empty test arrays as placeholders
                test_input_ids = np.array([]).reshape(0, 128)  # Empty but correct shape
                test_attention_mask = np.array([]).reshape(0, 128)
                logging.warning("Created empty test arrays due to no test data")
                logging.info(f"Arrays created - Train: {train_input_ids.shape}, Test: {test_input_ids.shape}")
            
            # Save arrays
            logging.info("Saving numpy arrays...")
            save_np_array(train_input_ids, self.data_transformation_config.data_transformation_train_input_ids)
            save_np_array(train_attention_mask, self.data_transformation_config.data_transformation_train_attention_mask)
            
            # Always save test arrays (even if empty) to maintain consistency
            save_np_array(test_input_ids, self.data_transformation_config.data_transformation_test_input_ids)
            save_np_array(test_attention_mask, self.data_transformation_config.data_transformation_test_attention_mask)
            
            logging.info("All arrays saved successfully!")
            
            # Log final statistics
            logging.info(f"Final statistics:")
            logging.info(f"  Train input_ids shape: {train_input_ids.shape}")
            logging.info(f"  Train attention_mask shape: {train_attention_mask.shape}")
            logging.info(f"  Test input_ids shape: {test_input_ids.shape}")
            logging.info(f"  Test attention_mask shape: {test_attention_mask.shape}")
            
            logging.info("Transformation completed successfully!")
            
            # Create artifact
            data_transformation_artifact = DataTransformationArtifact(
                train_attention_mask=self.data_transformation_config.data_transformation_train_attention_mask,
                train_input_id=self.data_transformation_config.data_transformation_train_input_ids,
                test_input_id=self.data_transformation_config.data_transformation_test_input_ids,
                test_attention_mask=self.data_transformation_config.data_transformation_test_attention_mask
            )
            
            return data_transformation_artifact
            
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)