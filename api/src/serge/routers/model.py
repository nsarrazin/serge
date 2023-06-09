import os
import urllib.request

import huggingface_hub
from fastapi import APIRouter, HTTPException

from serge.utils.convert import convert_one_file
from serge.utils.migrate import migrate

model_router = APIRouter(
    prefix="/model",
    tags=["model"],
)

models_info = {
    "GPT4AlpacaLoRA-30B": [
        "TheBloke/gpt4-alpaca-lora-30B-4bit-GGML",
        "gpt4-alpaca-lora-30b.ggmlv3.q5_1.bin",
        24.4e9,
    ],
    "AlpacaLoRA-65B": [
        "TheBloke/alpaca-lora-65B-GGML",
        "alpaca-lora-65B.ggmlv3.q5_1.bin",
        48.97e9,
    ],
    "OpenAssistant-30B": [
        "TheBloke/OpenAssistant-SFT-7-Llama-30B-GGML",
        "OpenAssistant-SFT-7-Llama-30B.ggmlv3.q5_1.bin",
        24.4e9,
    ],
    "GPT4All-13B": [
        "TheBloke/GPT4All-13B-snoozy-GGML",
        "GPT4All-13B-snoozy.ggmlv3.q5_1.bin",
        9.76e9,
    ],
    "StableVicuna-13B": [
        "TheBloke/stable-vicuna-13B-GGML",
        "stable-vicuna-13B.ggmlv3.q5_1.bin",
        9.76e9,
    ],
    "Vicuna-v1.1-7B": [
        "TheBloke/vicuna-7B-1.1-GGML",
        "vicuna-7b-1.1.ggmlv3.q5_1.bin",
        5.06e9,
    ],
    "Vicuna-v1.1-13B": [
        "TheBloke/vicuna-13b-1.1-GGML",
        "vicuna-13b-1.1.ggmlv3.q5_1.bin",
        9.76e9,
    ],
    "Vicuna-CoT-7B": [
        "TheBloke/Vicuna-7B-CoT-GGML",
        "vicuna-7B-cot.ggmlv3.q5_1.bin",
        5.06e9,
    ],
    "Vicuna-CoT-13B": [
        "TheBloke/Vicuna-13B-CoT-GGML",
        "vicuna-13b-cot.ggmlv3.q5_1.bin",
        9.76e9,
    ],
    "Guanaco-7B": [
        "TheBloke/guanaco-7B-GGML",
        "guanaco-7B.ggmlv3.q5_1.bin",
        5.06e9,
    ],
    "Guanaco-13B": [
        "TheBloke/guanaco-13B-GGML",
        "guanaco-13B.ggmlv3.q5_1.bin",
        9.76e9,
    ],
    "Guanaco-33B": [
        "TheBloke/guanaco-33B-GGML",
        "guanaco-33B.ggmlv3.q5_1.bin",
        24.4e9,
    ],
    "Guanaco-65B": [
        "TheBloke/guanaco-65B-GGML",
        "guanaco-65B.ggmlv3.q5_1.bin",
        48.97e9,
    ],
    "Wizard-Vicuna-Uncensored-7B": [
        "TheBloke/Wizard-Vicuna-7B-Uncensored-GGML",
        "Wizard-Vicuna-7B-Uncensored.ggmlv3.q5_1.bin",
        5.06e9,
    ],
    "Wizard-Vicuna-Uncensored-13B": [
        "TheBloke/Wizard-Vicuna-13B-Uncensored-GGML",
        "Wizard-Vicuna-13B-Uncensored.ggmlv3.q5_1.bin",
        9.76e9,
    ],
    "Wizard-Vicuna-Uncensored-30B": [
        "TheBloke/Wizard-Vicuna-30B-Uncensored-GGML",
        "Wizard-Vicuna-30B-Uncensored.ggmlv3.q5_1.bin",
        24.4e9,
    ],
    "WizardLM-30B": [
        "TheBloke/WizardLM-30B-GGML",
        "wizardlm-30b.ggmlv3.q5_1.bin",
        24.4e9,
    ],
    "WizardLM-Uncensored-7B": [
        "TheBloke/WizardLM-7B-uncensored-GGML",
        "WizardLM-7B-uncensored.ggmlv3.q5_1.bin",
        5.06e9,
    ],
    "WizardLM-Uncensored-13B": [
        "TheBloke/WizardLM-13B-Uncensored-GGML",
        "WizardLM-13B-Uncensored.ggmlv3.q5_1.bin",
        9.76e9,
    ],
    "WizardLM-Uncensored-30B": [
        "TheBloke/WizardLM-30B-Uncensored-GGML",
        "WizardLM-30B-Uncensored.ggmlv3.q5_1.bin",
        24.4e9,
    ],
    "Wizard-Mega-13B": [
        "TheBloke/wizard-mega-13B-GGML",
        "wizard-mega-13B.ggmlv3.q5_1.bin",
        9.76e9,
    ],
    "Lazarus-30B": [
        "TheBloke/30B-Lazarus-GGML",
        "30b-Lazarus.ggmlv3.q5_1.bin",
        24.4e9,
    ],
    "Nous-Hermes-13B": [
        "TheBloke/Nous-Hermes-13B-GGML",
        "nous-hermes-13b.ggmlv3.q5_1.bin",
        9.76e9,
    ],
    "Samantha-7B": [
        "TheBloke/Samantha-7B-GGML",
        "Samantha-7B.ggmlv3.q5_1.bin",
        5.06e9,
    ],
    "Samantha-13B": [
        "TheBloke/Samantha-13B-GGML",
        "samantha-13b.ggmlv3.q5_1.bin",
        9.76e9,
    ],
    "Samantha-33B": [
        "TheBloke/Samantha-33B-GGML",
        "samantha-33B.ggmlv3.q5_1.bin",
        24.4e9,
    ],
    "Koala-7B": [
        "TheBloke/koala-7B-GGML",
        "koala-7B.ggmlv3.q5_1.bin",
        5.06e9,
    ],
    "Koala-13B": [
        "TheBloke/koala-13B-GGML",
        "koala-13B.ggmlv3.q5_1.bin",
        9.76e9,
    ],
}

WEIGHTS = "/usr/src/app/weights/"


@model_router.get("/all")
async def list_of_all_models():
    res = []
    installed_models = await list_of_installed_models()
    for model in models_info.keys():
        progress = await download_status(model)
        if f"{model}.bin" in installed_models:
            available = True
            # if model exists in WEIGHTS directory remove it from the list
            installed_models.remove(f"{model}.bin")
        else:
            available = False
        res.append(
            {
                "name": model,
                "size": models_info[model][2],
                "available": available,
                "progress": progress,
            }
        )
    # append the rest of the models
    for model in installed_models:
        # .bin is removed for compatibility with generate.py
        res.append(
            {
                "name": model.replace(".bin", "").lstrip("/"),
                "size": os.stat(WEIGHTS + model).st_size,
                "available": True,
                "progress": None,
            }
        )

    return res


@model_router.get("/downloadable")
async def list_of_downloadable_models():
    files = os.listdir(WEIGHTS)
    files = list(filter(lambda x: x.endswith(".bin"), files))

    installed_models = [i.rstrip(".bin") for i in files]

    return list(filter(lambda x: x not in installed_models, models_info.keys()))


@model_router.get("/installed")
async def list_of_installed_models():
    # after iterating through the WEIGHTS directory, return location and filename
    files = [
        model_location.replace(WEIGHTS, "") + "/" + bin_file
        for model_location, directory, filenames in os.walk(WEIGHTS)
        for bin_file in filenames
        if os.path.splitext(bin_file)[1] == ".bin"
    ]
    files = [i.lstrip("/") for i in files]
    return files


@model_router.post("/{model_name}/download")
def download_model(model_name: str):
    models = list(models_info.keys())
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    if not os.path.exists(WEIGHTS + "tokenizer.model"):
        print("Downloading tokenizer...")
        url = huggingface_hub.hf_hub_url(
            "nsarrazin/alpaca",
            "alpaca-7B-ggml/tokenizer.model",
            repo_type="model",
            revision="main",
        )
        urllib.request.urlretrieve(url, WEIGHTS + "tokenizer.model")

    repo_id, filename, _ = models_info[model_name]

    print(f"Downloading {model_name} model from {repo_id}...")
    url = huggingface_hub.hf_hub_url(repo_id, filename, repo_type="model", revision="main")
    urllib.request.urlretrieve(url, WEIGHTS + f"{model_name}.bin.tmp")

    os.rename(WEIGHTS + f"{model_name}.bin.tmp", WEIGHTS + f"{model_name}.bin")
    convert_one_file(WEIGHTS + f"{model_name}.bin", WEIGHTS + "tokenizer.model")
    migrate(WEIGHTS + f"{model_name}.bin")

    return {"message": f"Model {model_name} downloaded"}


@model_router.get("/{model_name}/download/status")
async def download_status(model_name: str):
    models = list(models_info.keys())

    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    filesize = float(str(models_info[model_name][2]))

    bin_path = WEIGHTS + f"{model_name}.bin.tmp"

    if os.path.exists(bin_path):
        currentsize = os.path.getsize(bin_path)
        return min(round(currentsize / filesize * 100, 1), 100)
    return None


@model_router.delete("/{model_name}")
async def delete_model(model_name: str):
    if model_name + ".bin" not in await list_of_installed_models():
        raise HTTPException(status_code=404, detail="Model not found")

    if os.path.exists(WEIGHTS + f"{model_name}.bin"):
        os.remove(WEIGHTS + f"{model_name}.bin")
        return {"message": f"Model {model_name} deleted"}

    raise HTTPException(status_code=404, detail="Model file not found")
