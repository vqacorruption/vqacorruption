import json
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import datetime
import traceback
import torch
import requests
from PIL import Image
import torch

# from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
from difflib import SequenceMatcher
# from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
# from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

import ollama


class QwenOllamaVQAEvaluator:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)

        # Get Qwen-specific configuration
        self.model_config = self.config["open_source_models"]["qwenollama"]
        self.sampling_percentage = self.config.get("sampling_percentage", 100)
        self.unable_to_respond_aware = self.config.get("unable_to_respond_aware", True)
        self.initialize_model()

    def _create_prompt(self, question, ocr_text=None):
        unable_to_respond_line = "- If uncertain, return 'Unable to determine'\n- If you can't find the answer, return 'Unable to determine'" if self.unable_to_respond_aware else ""
        if ocr_text:
            return (
                f"You are an AI assistant specialized in analyzing document images and text. "
                f"Your task is to answer questions about the document image content precisely.\n\n"
                f"For this question, you have the following OCR text:\n{ocr_text}\n\n"
                f"Guidelines:\n"
                f"- Provide concise, focused answers (single word or short phrase)\n"
                f"- Base your answer on both the image and the provided OCR text\n"
                f"{unable_to_respond_line}\n"
                f"Question: {question}\n"
            )
        return (
            f"You are an AI assistant specialized in analyzing document images. "
            f"Your task is to answer questions about the document image content precisely.\n\n"
            f"Guidelines:\n"
            f"- Provide concise, focused answers (single word or short phrase)\n"
            f"- Base your answer exclusively on what you see in the image\n"
            f"{unable_to_respond_line}\n"
            f"Question: {question}"
        )

    def initialize_model(self):
        print("Initializing Qwen 72 Vision model...")
        print("Model configuration:", self.model_config)
    #     model_name = self.model_config["model_name"]
    #     self.model = Gemma3ForConditionalGeneration.from_pretrained(
    #         model_name, 
    #         device_map="auto",
    #         cache_dir="/data1/hf_cache/models"
    #     ).eval()

    #     self.processor = AutoProcessor.from_pretrained(model_name)

        self.max_tokens = self.model_config.get("max_tokens", 1024)
        print("Qwen 72 vision model initialized successfully")

    def _cleanup_model(self):
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()
            import gc

            gc.collect()

    def get_sorted_ocr_text(self, layout_analysis):
        """Extract and sort OCR text by bounding box position"""
        ocr_items = []
        for obj in layout_analysis.values():
            if isinstance(obj, dict) and "OCR" in obj and "BBOX" in obj:
                bbox = obj["BBOX"]
                ocr_items.append((bbox[1], bbox[0], obj["OCR"]))  # y, x, text

        # Sort by y coordinate first, then x coordinate
        ocr_items.sort()
        return "\n".join(item[2] for item in ocr_items)

    def generate_answer(self, question, image_paths, ocr_text=None):
        try:

            window_size = self.model_config.get("batch_size", 1)
            if window_size > 1:
                stride = self.model_config.get(
                    "stride", window_size // 2
                )  # Default stride is half the window size
            else:
                stride = 1
            total_images = len(image_paths)

            # Calculate total number of windows
            total_windows = max(1, (total_images - window_size + stride) // stride)

            # print(
            #     f"\nProcessing {total_images} images with window size {window_size}, "
            #     f"stride {stride} ({total_windows} window{'s' if total_windows > 1 else ''})"
            # )
            all_responses = []

            # Process images using moving window
            for window_idx in range(total_windows):
                start_idx = window_idx * stride
                end_idx = min(start_idx + window_size, total_images)
                window_paths = image_paths[start_idx:end_idx]

                # Handle last window to ensure all images are processed
                if window_idx == total_windows - 1 and end_idx < total_images:
                    window_paths = image_paths[-window_size:]
                batch_paths = window_paths

                # print(f"\nBatch {window_idx + 1}/{total_windows}")
                # print(f"Processing images {start_idx + 1}-{end_idx} of {total_images}")
                # print(f"Images in this batch: {len(window_paths)}")
                # for idx, path in enumerate(batch_paths, 1):
                #     print(f"  {idx}. {os.path.basename(path)}")

                # Get OCR text for this batch if available
                batch_ocr = None
                if ocr_text:
                    batch_ocr = []
                    for page_idx, path in enumerate(batch_paths, start_idx):
                        page_num = page_idx + 1
                        page_ocr = ocr_text.get(
                            path, ""
                        )  # Get OCR text for specific page
                        if page_ocr:
                            batch_ocr.append(f"Page {page_num}:\n{page_ocr}")
                    batch_ocr = "\n\n".join(batch_ocr) if batch_ocr else None


                # Create prompt for the batch
                question_prompt = self._create_prompt(question, batch_ocr)
                # print("Batch length: ", len(batch_paths))
                # inputs = self.processor.process(
                #     images=[Image.open(path).convert("RGB") for path in batch_paths],
                #     text=question_prompt,
                # )
                # inputs = {
                #     k: v.to("cuda:0").unsqueeze(0) for k, v in inputs.items()
                # }
                # images=[Image.open(path).convert("RGB") for path in batch_paths]
                output = ollama.chat(
                    model='qwen2.5vl:72b-q8_0',
                    messages=[{
                        'role': 'user',
                        'content': question_prompt,
                        'images': batch_paths
                    }]
                )
                response=output['message']['content']
                # print(output)
                # print(response)
                # print(response)

                # print(f"Response for batch {window_idx + 1}: {response}")

                # Add response for this batch
                all_responses.append(
                    {
                        "pages": batch_paths,  # Using actual image paths for this batch
                        "answer": response,
                    }
                )

            return {
                "answer": all_responses,
                "query": question,
                "image_paths": image_paths,
                # "analysis_type": f"batch_size_{batch_size}",
                "analysis_type": f"window_size_{window_size}",
            }

        except Exception as e:
            print(f"Error in generate_answer: {str(e)}")
            print(f"Full error: {traceback.format_exc()}")
            return {
                "answer": "Unable to determine: error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _save_results(self, data):
        output_file = self.model_config["name"]+"_"+self.config["output_file"]
        if self.config["ocr_enabled"] and not self.config["unable_to_respond_aware"]:
            output_file = output_file.replace(".json", "_OCR_UNABLE.json")
        elif self.config["ocr_enabled"]:
            output_file = output_file.replace(".json", "_OCR.json")
        elif not self.config["unable_to_respond_aware"]:
            output_file = output_file.replace(".json", "_UNABLE.json")
        try:
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Results successfully saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def evaluate(self):
        try:
            print("\nStarting Gemma evaluation...")

            # Load input data
            with open(self.config["input_file"]) as f:
                data = json.load(f)
                print(f"Successfully loaded input file: {self.config['input_file']}")

            # Sample questions
            total_questions = len(data["corrupted_questions"])
            num_samples = int(total_questions * (self.sampling_percentage / 100))

            if self.sampling_percentage < 100:
                sampled_questions = random.sample(
                    data["corrupted_questions"], num_samples
                )
                data["corrupted_questions"] = sampled_questions
                print(
                    f"Sampled {num_samples} questions ({self.sampling_percentage}%) for evaluation"
                )
            else:
                print("Processing 100% of questions (no sampling)")

            processed_count = 0
            success_count = 0
            error_count = 0

            for item in tqdm(data["corrupted_questions"]):
                try:
                    processed_count += 1
                    # print(f"\nProcessing question {processed_count}/{len(data['corrupted_questions'])}")

                    if "verification_result" not in item:
                        item["verification_result"] = {}
                    if "vqa_results" not in item["verification_result"]:
                        item["verification_result"]["vqa_results"] = []

                    question = item["corrupted_question"]
                    pages = item["layout_analysis"]["pages"]

                    image_paths = []
                    for page_id in pages:
                        image_filename = os.path.basename(page_id)
                        image_path = os.path.join(
                            self.config["images_base_path"], image_filename
                        )
                        image_paths.append(image_path)

                    # print(f"Question: {question[:100]}...")
                    # print(f"Number of pages to analyze: {len(image_paths)}")

                    # Get OCR text if enabled
                    ocr_text = None
                    if self.config.get("ocr_enabled", False):
                        # print("Extracting OCR text...")
                        ocr_text = {}
                        for page_id in pages:
                            # Navigate through the nested structure correctly
                            page_layout = pages[page_id]["layout_analysis"]
                            page_ocr = self.get_sorted_ocr_text(page_layout)
                            if page_ocr:
                                # Use the full path as the key since that's what we use in generate_answer
                                image_filename = os.path.basename(page_id)
                                image_path = os.path.join(
                                    self.config["images_base_path"], image_filename
                                )
                                ocr_text[image_path] = page_ocr
                                # print(f"Extracted OCR text for page: {image_filename}")
                            else:
                                print(f"No OCR text found for page: {image_filename}")

                    # Generate answer
                    # print("Generating answer...")
                    result = self.generate_answer(question, image_paths, ocr_text)
                    # print(f"\nAnswer received: {result.get('answer', 'No answer')}")

                    # Create VQA result
                    vqa_result = {
                        "model_type": "qwen",
                        "model_config": {
                            "batch_size": self.model_config.get("batch_size", 1),
                            "max_tokens": self.max_tokens,
                            "use_flash_attention": self.model_config.get(
                                "use_flash_attention", False
                            ),
                        },
                        "ocr_enabled": bool(ocr_text),
                        "question": question,
                        "answer": result.get("answer", "Unable to determine"),
                        "image_paths": image_paths,
                        "analysis_type": result.get("analysis_type", ""),
                        "timestamp": datetime.datetime.now().isoformat(),
                    }

                    if "error" in result:
                        vqa_result["error"] = result["error"]
                        vqa_result["traceback"] = result.get("traceback", "")
                        error_count += 1
                    else:
                        success_count += 1

                    item["verification_result"]["vqa_results"].append(vqa_result)

                    # Save intermediate results
                    # if processed_count % 10 == 0:
                    #     self._save_results(data)
                    #     print("Intermediate results saved")

                except Exception as e:
                    print(f"Error processing question: {str(e)}")
                    print(f"Full error: {traceback.format_exc()}")
                    error_count += 1

            # Final statistics
            print(f"\nProcessing completed:")
            print(f"Total questions processed: {processed_count}")
            print(f"Successful generations: {success_count}")
            print(f"Errors encountered: {error_count}")
            if processed_count > 0:
                print(f"Success rate: {(success_count/processed_count)*100:.2f}%")

            # Save final results
            self._save_results(data)
            output_file = self.model_config["name"]+"_"+self.config["output_file"]
            if self.config["ocr_enabled"] and not self.config["unable_to_respond_aware"]:
                output_file = output_file.replace(".json", "_OCR_UNABLE.json")
            elif self.config["ocr_enabled"]:
                output_file = output_file.replace(".json", "_OCR.json")
            elif not self.config["unable_to_respond_aware"]:
                output_file = output_file.replace(".json", "_UNABLE.json")
            print(f"Final results saved to {output_file}")

        except Exception as e:
            print(f"Critical error in evaluate: {str(e)}")
            print(f"Full error: {traceback.format_exc()}")
        finally:
            self._cleanup_model()


def main():
    config_path = "../config.json"  # Update this path to match your config file location
    evaluator = QwenOllamaVQAEvaluator(config_path)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
