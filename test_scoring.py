# test_scoring_colintern.py
from pathlib import Path
from PIL import Image
import torch
import random
import numpy as np
from colpali_engine.models.internvl3_5.colintern3_5 import ColIntern3_5_Processor, ColIntern3_5

# ---------- Config ----------
MODEL_DIR = "./outputs/colintern3_5-run1/checkpoint-1847"
PROC_DIR  = "./outputs/colintern3_5-run1"

model = ColIntern3_5.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to("cuda").eval()
IMAGE_ROOT = "."  # change if your images live elsewhere
IMAGE_FILES = {
    "pepe":      str(Path(IMAGE_ROOT) / "pepe.png"),
    "explorer":  str(Path(IMAGE_ROOT) / "explorer.png"),
    "colintern": str(Path(IMAGE_ROOT) / "colintern.png"),
}

# ---------- Determinism ----------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

# ---------- Load ----------
model = ColIntern3_5.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to("cuda").eval()

processor = ColIntern3_5_Processor.from_pretrained(PROC_DIR)
print(f"[model] device={next(model.parameters()).device}, dtype={next(model.parameters()).dtype}")

model_dtype = next(model.parameters()).dtype
model_device = next(model.parameters()).device
print(f"[model] device={model_device}, dtype={model_dtype}")

# Helper to ensure we don't duplicate POS as a negative
def _unique_negs(pos_path, neg_paths):
    pos_name = Path(pos_path).name
    return [p for p in neg_paths if Path(p).name != pos_name]

def rank_once(query: str, pos_path: str, neg_paths: list[str]):
    """Ranks [pos] + negs for one query. Prints names + scores and returns top1_is_pos, margin."""
    neg_paths = _unique_negs(pos_path, neg_paths)

    # instruction prefix (matches ModelMeta.use_instructions=True)
    query_prefixed = f"{getattr(processor, 'query_prefix', 'Query: ')}{query}"

    # --- preprocess ---
    q_batch = processor.process_texts([query_prefixed])
    p_imgs = [Image.open(pos_path)] + [Image.open(p) for p in neg_paths]
    p_batch = processor.process_images(p_imgs)

    q_ids  = q_batch["input_ids"].to(model_device)
    q_mask = q_batch["attention_mask"].to(model_device)
    pix    = p_batch["pixel_values"].to(model_device)  # forward() will cast to model dtype

    # --- embed & score ---
    with torch.no_grad():
        qs = model.embed_queries(q_ids, q_mask)           # (1, M, 128)
        Ps = model(pixel_values=pix)                       # safe path for dtype/device
        scores = processor.score_multi_vector(qs, Ps)[0]   # (1+K,)

    names = ["POS"] + [f"NEG:{Path(p).name}" for p in neg_paths]
    order = scores.argsort(descending=True).tolist()

    print("\n--- Query:", repr(query_prefixed))
    print("Ranked docs (best → worst):")
    for i in order:
        print(f"{i:2d}  {names[i]:18s}  score={float(scores[i].detach()):.4f}")

    top1_is_pos = (order[0] == 0)
    margin = float(scores[order[0]].detach() - scores[0].detach())  # how far top beats POS
    print(f"Top-1 is POS? {top1_is_pos}   (margin vs POS = {margin:.4f})")

    # quick health checks
    q_valid = int(q_mask.sum().item())
    q_norm = qs[q_mask.bool()].norm(dim=-1).mean().item() if q_valid else float('nan')
    d_norm = Ps.norm(dim=-1).mean().item()
    print(f"valid query tokens: {q_valid} | mean ||q|| ≈ {q_norm:.3f} | mean ||d|| ≈ {d_norm:.3f}")

    return top1_is_pos, margin

def run_suite():
    cases = [
        # Pepe (cartoon frog with sunglasses)
        ("cartoon image of 'Pepe the Frog' wearing black sunglasses; meme style",
         IMAGE_FILES["pepe"],
         [IMAGE_FILES["explorer"], IMAGE_FILES["colintern"]]),

        # Explorer panel (comic caption, helmet lamp, green rabbit key)
        ("vintage comic panel of a cave explorer with helmet lamp holding a glowing green rabbit-shaped key in front of a door with a rabbit keyhole; caption present",
         IMAGE_FILES["explorer"],
         [IMAGE_FILES["pepe"], IMAGE_FILES["colintern"]]),

        # ColIntern poster (muscular mascot, cape, title text)
        ("golden poster of a muscular superhero mascot flexing biceps with a cape; big title text 'COLINTERN' at the top",
         IMAGE_FILES["colintern"],
         [IMAGE_FILES["pepe"], IMAGE_FILES["explorer"]]),

        # Alternative phrasings (helps probe robustness)
        ("meme of a green cartoon frog wearing sunglasses (Pepe)",
         IMAGE_FILES["pepe"],
         [IMAGE_FILES["explorer"], IMAGE_FILES["colintern"]]),

        ("comic scene of an explorer in a cave opening a door using a glowing rabbit totem",
         IMAGE_FILES["explorer"],
         [IMAGE_FILES["pepe"], IMAGE_FILES["colintern"]]),

        ("superhero mascot bodybuilder flexing with cape; audience cheering; title COLINTERN",
         IMAGE_FILES["colintern"],
         [IMAGE_FILES["pepe"], IMAGE_FILES["explorer"]]),
    ]

    ok = 0
    for q,pos,negs in cases:
        top1, _ = rank_once(q, pos, negs)
        ok += int(top1)
    print(f"\n=== Mini accuracy: {ok}/{len(cases)} top-1 on obvious pairs ===")

if __name__ == "__main__":
    # Ensure files exist
    for k, p in IMAGE_FILES.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing image for '{k}': {p}")
    run_suite()
