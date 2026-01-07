# Fine-Grained Visual Contrast-Driven Preference Optimization for Mitigating Hallucinations in LVLMs

This repository is provided for anonymous peer review.

## Environment Setup

To set up the environment for this project, follow these steps:

```bash
conda create -n fgvpo python=3.10
conda activate fgvpo
pip install -r requirements.txt
```

## Download Base Models

Create a folder `base_models` and download the required models, `llava-v1.5-7b` and `vision_tower-clip336`, from the web.

```bash
# Create the base_models directory
mkdir -p /path/to/your/project/base_models

# Download the llava-v1.5-7b model
wget https://huggingface.co/llava/llava-v1.5-7b/resolve/main/llava-v1.5-7b.bin -P /path/to/your/project/base_models/

# Download the vision_tower-clip336 model
wget https://huggingface.co/llava/vision_tower-clip336/resolve/main/vision_tower-clip336.bin -P /path/to/your/project/base_models/
```

## Merge FG-VPO Model
To merge the FG-VPO model, you can download our model's checkpoint from https://huggingface.co/anonymous260105/FG-VPO/tree/main
 and run the merge script. This process combines the llava-v1.5-7b model with the FG-VPO adapter for hallucination mitigation.
 
```bash
# Download FG-VPO checkpoint
wget https://huggingface.co/anonymous260105/FG-VPO/resolve/main/adapter_config.json -P /path/to/your/project/checkpoint-final/adapter_model/lora_policy

wget https://huggingface.co/anonymous260105/FG-VPO/resolve/main/adapter_model.bin -P /path/to/your/project/checkpoint-final/adapter_model/lora_policy

# Merge the base model with the FG-VPO adapter
python run/merge/merge_llava_lora.py \
  --model-path ./base_models/llava-v1.5-7b \
  --lora-model-path ./output/llava_fgvpo/checkpoint-final/adapter_model/lora_policy \
  --save-model-path ./output/llava_fgvpo/llava_fgvpo_merged
```

## Evaluation

### Download Evaluation Data1 (POPE, AMBER, TextVQA, and MMBench)

We have adopted the four evaluation datasets from **LLaVA**, including **POPE**, **AMBER**, **TextVQA**, and **MMBench**. These datasets are provided in the `eval.zip` file, which was also used in **LLaVA**.

Download the `eval.zip` file from the following link:

[Download eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view)

After downloading, extract the contents of the zip file to the `evaluation` folder.

```bash
# Create the `evaluation` directory**
mkdir -p /path/to/your/project/evaluation

# After downloading `eval.zip`, unzip it and move its content to the `evaluation` directory 
unzip eval.zip
mv /path/to/your/downloaded/eval /path/to/your/project/evaluation/
cd /path/to/your/project/evaluation/
```

### Evaluation1
```bash
cd /path/to/your/project/run/eval/pope
bash pope.sh

cd /path/to/your/project/run/eval/amber
bash amber.sh

cd /path/to/your/project/run/eval/textvqa
bash textvqa.sh

cd /path/to/your/project/run/eval/mmbench
bash mmbench.sh
```
### Download Evaluation Data2 (CHAIR)

To evaluate hallucinations in object captioning, use the **CHAIR** metric. You can download the CHAIR evaluation script from the following link:

- **CHAIR GitHub Repository**: [https://github.com/LisaAnne/Hallucination](https://github.com/LisaAnne/Hallucination)


```bash
# Clone the CHAIR repository
git clone https://github.com/LisaAnne/Hallucination.git

# Move chair.py to the evaluation directory
mv Hallucination/utils/chair.py /path/to/your/project/evaluation/chair
```

- **Download MSCOCO annotations**:
```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P /path/to/your/project/evaluation/MSCOCO

# After downloading, extract the annotations to the `evaluation` folder
unzip annotations_trainval2014.zip
```

### Evaluation2
```bash
cd /path/to/your/project/run/eval/chair
bash chair.sh
```

Additional training details and datasets will be shared after the paper is accepted for publication.
