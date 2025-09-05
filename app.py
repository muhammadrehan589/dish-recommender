import streamlit as st
import pandas as pd
import joblib
from rapidfuzz import process, fuzz

# MUST be the first Streamlit command in the script
st.set_page_config(page_title="Dish Recommender", page_icon="🍲", layout="centered")

# --- Auto-download models from Google Drive if missing ---
import os
import sys

try:
    import gdown
except Exception:
    gdown = None

# Google Drive file IDs (from links you provided)
FILES = {
    "models/tfidf_ing.joblib": "1vp8EdbGFJfPhdeJ5hu_MnHc7IXNkVOtR",
    "models/nn_ing.joblib": "1_f7DJdWcxon5IoR0-n4SEWg7-FaTDpLr",
    "models/recipes_meta.pkl": "1MHXvJxgu2huKpGLtUoezZktmDOHkgL7b",
}

os.makedirs("models", exist_ok=True)

def download_file(file_id, output_path):
    if os.path.exists(output_path):
        return
    if gdown is None:
        print(f"gdown not installed. Install with: pip install gdown", file=sys.stderr)
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {output_path} from Drive id={file_id}...", file=sys.stderr)
    gdown.download(url, output_path, quiet=False)

for out_path, fid in FILES.items():
    if not fid:
        print(f"Missing Drive ID for {out_path}", file=sys.stderr)
    else:
        download_file(fid, out_path)

# -----------------------------
# Load models and data
# -----------------------------
@st.cache_resource
def load_model():
    try:
        tfidf = joblib.load("models/tfidf_ing.joblib")
        nn = joblib.load("models/nn_ing.joblib")
        meta = pd.read_pickle("models/recipes_meta.pkl")

        # normalize vocabulary and metadata ingredients to match prep_user_ings
        vocab = tfidf.get_feature_names_out().tolist()

        def _norm_token(t):
            return t.strip().lower().replace(" ", "_")

        # ensure ingredients_parsed exists and create a normalized version
        if "ingredients_parsed" in meta.columns:
            meta["ingredients_parsed_norm"] = meta["ingredients_parsed"].apply(
                lambda lst: [_norm_token(x) for x in lst]
            )
        else:
            meta["ingredients_parsed_norm"] = [[] for _ in range(len(meta))]

        # also make sure steps_parsed exists (unchanged)
        if "steps_parsed" not in meta.columns:
            meta["steps_parsed"] = [[] for _ in range(len(meta))]

        return tfidf, nn, meta, vocab
    except Exception as e:
        # print to terminal and show UI error
        print("Error loading models:", e)
        st.error(f"Error loading models: {e}")
        return None, None, None, []

tfidf, nn, meta, VOCAB = load_model()

# stop early if loading failed
if tfidf is None or nn is None or meta is None:
    st.error(
        "Models failed to load. Check the terminal for errors and ensure models/ contains "
        "tfidf_ing.joblib, nn_ing.joblib, and recipes_meta.pkl"
    )
    st.stop()

# -----------------------------
# Helper Functions
# -----------------------------
def prep_user_ings(ings):
    def norm(s):
        return s.strip().lower().replace("/", " ").replace("-", " ").replace("  ", " ").strip()

    tokens = [norm(i) for i in ings]
    snapped = []

    for t in tokens:
        t_underscore = t.replace(" ", "_")

        # 1) exact normalized vocab hit
        if t_underscore in VOCAB:
            snapped.append(t_underscore)
            continue

        words = [w for w in t.split() if w]

        # 2) prefer vocab entries that contain all token words (as whole or substring)
        candidates = []
        for v in VOCAB:
            v_words = v.split("_")
            if all(any(w == vw or w in vw or vw in w for vw in v_words) for w in words):
                candidates.append(v)
        if candidates:
            best_cand = max(candidates, key=lambda c: fuzz.token_sort_ratio(t_underscore, c))
            score = fuzz.token_sort_ratio(t_underscore, best_cand)
            if score >= 60:
                snapped.append(best_cand)
                continue

        # 3) heuristic for stock/broth variants: try to find vocab containing the main noun + stock/broth
        if any(x in t for x in ("stock", "broth")) and words:
            main = words[0]
            for v in VOCAB:
                if main in v and ("stock" in v or "broth" in v):
                    snapped.append(v)
                    break
            else:
                # try underscored form
                snapped.append(t_underscore)
            continue

        # 4) global fuzzy fallback but require very high confidence to avoid bad matches
        best = process.extractOne(t_underscore, VOCAB, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= 85:
            snapped.append(best[0])
            continue

        # 5) otherwise keep the safe underscored token (no risky snap)
        snapped.append(t_underscore)

    return snapped

def recommend(ings, k=5, min_cov=0.4):
    if tfidf is None or nn is None or meta is None:
        return []

    user_tokens = prep_user_ings(ings)

    # token-level debug matches
    token_matches = []
    for t in user_tokens:
        best = process.extractOne(t, VOCAB, scorer=fuzz.token_sort_ratio)
        token_matches.append({"token": t, "best": best[0] if best else None, "score": best[1] if best else None})

    q = " ".join(user_tokens)
    qv = tfidf.transform([q])
    dists, idxs = nn.kneighbors(qv, n_neighbors=200)
    have = set(user_tokens)

    results = []
    for d, i in zip(dists[0], idxs[0]):
        r = meta.iloc[i]
        rec_set = set(r["ingredients_parsed_norm"])
        cov = len(have & rec_set) / max(1, len(rec_set))
        if cov >= min_cov:
            needs_norm = list(rec_set - have)[:5]
            needs_friendly = [n.replace("_", " ") for n in needs_norm]
            results.append({
                "name": r["name"],
                "coverage": cov,
                "needs": needs_friendly,
                "minutes": r.get("minutes", "unknown"),
                "steps": r["steps_parsed"][:5],
                # debug fields
                "debug_user_tokens": user_tokens,
                "debug_token_matches": token_matches,
                "debug_rec_set": list(rec_set),
                "debug_intersection": list(have & rec_set),
                "debug_missing_norm": needs_norm
            })
        if len(results) >= k:
            break
    return results

# -----------------------------
# Streamlit Web App (clean UI)
# -----------------------------
st.title("🍲 AI Dish Recommender")
st.write("Enter the ingredients you have, and I’ll suggest recipes with steps!")

ingredients = st.text_input("Your ingredients (comma separated):", "chicken, onion, garlic, rice")

if st.button("Find Recipes"):
    input_list = [i.strip() for i in ingredients.split(",")]

    recs = recommend(input_list, k=5)

    if not recs:
        st.warning("No recipes found. Try adding more ingredients!")
    else:
        for r in recs:
            st.subheader(f"🍽 {r['name']} ({int(r['coverage']*100)}% match)")
            st.write(f"⏱ Estimated time: {r['minutes']} minutes")

            if r["needs"]:
                st.write("🛒 Missing:", ", ".join(r["needs"]))
            st.markdown("**Steps:**")
            for s in r["steps"]:
                st.write(f"- {s}")
            st.markdown("---")



