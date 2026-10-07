import html, json, os, pathlib, re, subprocess, sys
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
C_TEXT, C_G, C_CC = "#4C8BF5", "#F5A623", "#2ECC71"

# Held-out test papers for the demo: loaded from data/subset.jsonl by AMiner id, so the
# title/abstract/year and the reference list are the real ones from the dataset.
HELD_OUT = {
    "WWW 2012 - Joint Relevance and Freshness Learning (news search)": "5390b4c420f70186a0efe9e8",
}

EXAMPLE = {
    "title": "Neural ranking models for ad-hoc retrieval",
    "year": 2016,
    "abstract": "We propose a deep neural network for ad-hoc document retrieval. "
                "The model learns a representation of queries and documents and ranks "
                "documents by relevance. Experiments on standard test collections show "
                "improvements over language model and BM25 baselines.",
}

# ---------------- backend: runs the CLI and parses its output ----------------
def parse(out):
    results, cur = [], None
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\.\s*\[([\d.]+)\]\s*\((\w+)\)\s*(.*)", line)
        if m:
            cur = {"rank": int(m[1]), "score": float(m[2]), "year": m[3],
                   "title": m[4].rstrip("."), "text": 0.0, "g": 0.0, "cc": 0.0, "why": []}
            results.append(cur)
        elif cur and "score =" in line:
            for key, pat in [("text", r"text\s*([\d.]+)"), ("g", r"g\(d\)\s*([\d.]+)"),
                             ("cc", r"co-citation\s*([\d.]+)")]:
                p = re.search(pat, line)
                if p:
                    cur[key] = float(p[1])
        elif cur and line.strip().startswith("why:"):
            cur["why"] = [(t, int(p)) for t, p in re.findall(r"([^\s,()]+)\s*\((\d+)%\)", line)]
        elif cur and "AMiner id" in line:
            cur["id"] = line.rsplit("AMiner id", 1)[1].strip()
    return results

def internals(out):
    """The --verbose block of the CLI output (preprocessing, ltc weights, postings)."""
    m = re.search(r"^=+\n(.*?)\n=+$", out, re.S | re.M)
    return m[1] if m else ""

def run_search(abstract, title, year, k, baseline=False, verbose=False):
    cmd = [sys.executable, "-m", "src.search", "--abstract", abstract, "-k", str(k)]
    if baseline:
        # ablation step 1: tf-idf on the abstract zone only, no year filter, no g(d), no co-citation
        cmd += ["--title-w", "0", "--g-weight", "0", "--cc-weight", "0"]
    else:
        if title:
            cmd += ["--title", title]
        if year:
            cmd += ["--year", str(year)]
    if verbose:
        cmd += ["--verbose"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", env=env)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "Search failed")
    return parse(p.stdout), p.stdout

@st.cache_data
def corpus():
    """AMiner id -> paper, for the held-out papers' real reference lists."""
    with open(ROOT / "data" / "subset.jsonl", encoding="utf-8") as f:
        return {p["id"]: p for p in map(json.loads, f)}

# ---------------- UI ----------------
st.set_page_config(page_title="Citation Finder", page_icon="📚", layout="centered",
                   initial_sidebar_state="collapsed")
st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl']"
            "{display:none}</style>", unsafe_allow_html=True)

def fill_example():
    st.session_state.title = EXAMPLE["title"]
    st.session_state.year = EXAMPLE["year"]
    st.session_state.abstract = EXAMPLE["abstract"]
    st.session_state.held_out = "-"

def fill_held_out():
    pid = HELD_OUT.get(st.session_state.held_out)
    if pid:
        p = corpus()[pid]
        st.session_state.title, st.session_state.year, st.session_state.abstract = p["title"], p["year"], p["abstract"]

st.title("📚 Citation Finder")
st.write("Paste a paper's abstract and get a ranked list of papers it should probably cite.")

c1, c2, c3 = st.columns([5, 1.5, 1.5])
c1.text_input("Title (optional)", key="title")
c2.number_input("Year (optional)", 0, 2100, 0, key="year", help="0 = no year filter")
k = c3.number_input("Results", 5, 50, 10, step=5)

st.text_area("Abstract", key="abstract", height=200,
             placeholder="Paste the abstract here...")

st.selectbox("Or load a held-out test paper (its real references are marked in the results)",
             ["-", *HELD_OUT], key="held_out", on_change=fill_held_out)
o1, o2 = st.columns(2)
baseline = o1.checkbox("Baseline: tf-idf on abstract only",
                       help="No title zone, no year filter, no g(d), no co-citation (ablation step 1)")
verbose = o2.checkbox("Show IR internals", help="Preprocessed query, ltc query weights, postings, lnc weights")

b1, b2, _ = st.columns([1, 1.3, 5])
go = b1.button("Search", type="primary", use_container_width=True)
b2.button("Try example", on_click=fill_example, use_container_width=True)

if go:
    if not st.session_state.abstract.strip():
        st.warning("Please enter an abstract.")
    else:
        try:
            with st.spinner("Searching... (loading the index takes a few seconds)"):
                results, out = run_search(st.session_state.abstract, st.session_state.title,
                                          st.session_state.year, k, baseline, verbose)
        except Exception as e:
            st.error("Search failed.")
            st.code(str(e))
            results, out = [], ""
        if verbose and internals(out):
            with st.expander("IR internals: preprocessing, ltc query weights, postings (docID:tf -> lnc weight)",
                             expanded=True):
                st.code(internals(out), language=None)
        # real reference list of the loaded held-out paper (only those inside our corpus)
        pid = HELD_OUT.get(st.session_state.get("held_out", "-"))
        refs = {r for r in corpus()[pid]["references"] if r in corpus()} if pid else set()
        if results:
            st.subheader(f"{len(results)} suggested citations"
                         + (" (baseline)" if baseline else ""))
            if refs:
                hits = sum(r.get("id") in refs for r in results)
                st.markdown(f"**{hits} of {len(results)}** are in this paper's real reference list "
                            f"({len(refs)} of its references are in our corpus).")
            st.markdown(
                f"<span style='color:{C_TEXT}'>■</span> text match &nbsp; "
                f"<span style='color:{C_G}'>■</span> citation count g(d) &nbsp; "
                f"<span style='color:{C_CC}'>■</span> co-citation", unsafe_allow_html=True)
            for r in results:
                parts = r["text"] + r["g"] + r["cc"] or 1
                bar = "".join(
                    f"<div style='width:{v / parts * 100:.1f}%;background:{c}'></div>"
                    for v, c in [(r["text"], C_TEXT), (r["g"], C_G), (r["cc"], C_CC)])
                chips = " ".join(
                    f"<span style='background:#262B3A;border-radius:12px;padding:2px 10px;"
                    f"margin-right:4px;font-size:0.85em'>{html.escape(t)} {p}%</span>"
                    for t, p in r["why"])
                with st.container(border=True):
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between'>"
                        f"<div><b>{r['rank']}. {html.escape(r['title'])}</b> "
                        f"<span style='opacity:.6'>({r['year']})</span>"
                        + (f" <span style='color:{C_CC};font-weight:600'>&#10003; cited by this paper</span>"
                           if refs and r.get("id") in refs else "")
                        + "</div>"
                        f"<div style='font-size:1.3em'><b>{r['score']:.3f}</b></div></div>"
                        f"<div style='display:flex;height:8px;border-radius:4px;overflow:hidden;"
                        f"margin:10px 0 4px'>{bar}</div>"
                        f"<div style='font-size:0.8em;opacity:.7'>text {r['text']:.3f} + "
                        f"g(d) {r['g']:.3f} + co-citation {r['cc']:.3f}</div>"
                        f"<div style='margin-top:8px'><span style='opacity:.7'>Why:</span> {chips}</div>",
                        unsafe_allow_html=True)
                    