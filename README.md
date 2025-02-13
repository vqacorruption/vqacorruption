<div align="center">
  
## Benchmarking Visual LLMs Robustness to Corrupted Questions in Multi-Page Document Visual Question Answering

### Under Review at ACL 2025
</div>
This study investigates the robustness of Multimodal Models (VLLMs and LLMs) to corrupted questions in multi-page document Visual Question Answering (DocVQA), aiming to create a comprehensive benchmark for evaluating model resilience across different types of input perturbations.
We evaluate their performance degradation when handling various levels of question corruption,
[TODO]

## Datasets

|                   | MPDocVQA | DUDE |
|-------------------|:--------:|:----:|
| Full              | [link*](https://rrc.cvc.uab.es/?ch=17&com=downloads) | [link*](https://rrc.cvc.uab.es/?ch=23&com=downloads) |
| Reduced           | [link]() | [link]() |
| Corrupted         | [link]() | [link]() |
| Verified          | [link]() | [link]() |

\* Link to original dataset repository

The "Reduced" row contains:
- the subsets of questions taken from the full datasets,
- the OCR and Layout analysis
- the dataset augmented
  
The "Corrupted" row contains corrupted questions.<br />
The "Verified" row contain the corrupted question verified by the Judge model.

In future, we will release the full datasets processed.


## Models
|                   | Type | Size | License      | Link |
|-------------------|:----:|:----:|:------------:|:----:|
| QWEN 2.5-VL       | VLLM | 8B   | Apache 2.0   |  [link](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| InternVL 2.5      | VLLM | 8B   | MIT          |  [link](https://huggingface.co/OpenGVLab/InternVL2_5-8B) |
| Phi 3.5 Vision    | VLLM | 4B   | MIT          |  [link](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) |
| Molmo             | VLLM | 8B   | Apache 2.0   |  [link](https://huggingface.co/allenai/Molmo-7B-D-0924) |
| Ovis              | VLLM | 10B  | Apache 2.0   |  [link](https://huggingface.co/AIDC-AI/Ovis1.6-Gemma2-9B) |
| DocOwl2           | VLLM | 8B   | Apache 2.0   |  [link](https://huggingface.co/mPLUG/DocOwl2) |
| UDOP              | VLM  | 742M | MIT          |  [link](https://huggingface.co/microsoft/udop-large) |
| LayoutLMv3        | VLM  | 125M | Llama-2      |  [link](https://huggingface.co/rubentito/layoutlmv3-base-mpdocvqa) |
| BLIP              | VLM  | 385M | bsd-3-clause |  [link](https://huggingface.co/Salesforce/blip-vqa-base) |


Gemini 2.0 Flash has been used as judge on all verification steps

After each model is tested, it is postprocess to standardize results.
The final output file has the same verified dataset elements extended with models answers.
In detail it has a list with:
-  document pages taken in the window to provide an answer
-  model answer
-  standardize model answer

## Execution

[TODO]

To run the experiments, install the dependencies in the `requirements.txt` file.<br />
Currently, the experiments have been executed with `transformers==4.49.0.dev0`. Due to compatibility issue, we suggest to downgrade to `transformers==4.48.2` when testing Phi 3.5.

## License
This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

## Citation

If you find this project useful, please consider citing:
```bibtex
```
<!--
**vqacorruption/vqacorruption** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
