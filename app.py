import os
import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array  # type: ignore
import numpy as np
from keras.models import load_model
import requests

# ─────────────────────────── Page config ────────────────────────────────────
st.set_page_config(
    page_title="FOODOTO | Food Recognizer",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #f2f7f2;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #1a3d2b;
    border-right: 1px solid #224d36;
}
section[data-testid="stSidebar"] * { color: #a8ccb4 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #d4edd9 !important; font-weight: 500 !important; }
section[data-testid="stSidebar"] .stMarkdown p { font-size: 0.84rem; line-height: 1.7; }
section[data-testid="stSidebar"] hr { border-color: #2a5a3e !important; }

/* ── Hero ── */
.hero {
    background-color: #1a3d2b;
    border-radius: 16px;
    padding: 40px 44px;
    margin-bottom: 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hero-text {}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: #e8f5ec;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.hero h1 span { color: #7ecf8e; }
.hero p {
    font-size: 0.92rem;
    color: #7aaa8a;
    margin: 0;
    max-width: 440px;
    line-height: 1.7;
    font-weight: 300;
}
.hero-badge {
    background-color: #224d36;
    border: 1px solid #2e6348;
    border-radius: 12px;
    padding: 16px 24px;
    text-align: center;
    min-width: 120px;
}
.hero-badge .num {
    font-size: 2rem;
    font-weight: 600;
    color: #7ecf8e;
    line-height: 1;
}
.hero-badge .sub {
    font-size: 0.7rem;
    color: #5a8a6a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Stat row ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 28px;
}
.stat-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid #d4e8d8;
    display: flex;
    align-items: center;
    gap: 14px;
}
.stat-icon {
    width: 40px; height: 40px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
}
.stat-icon.g { background: #d4edda; }
.stat-icon.b { background: #c8e6d0; }
.stat-icon.o { background: #e0f0e4; }
.stat-icon.p { background: #b8dfc4; }
.stat-val { font-size: 1.5rem; font-weight: 600; color: #1a3d2b; line-height: 1; }
.stat-lbl { font-size: 0.72rem; color: #6a9a7a; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

/* ── Upload box ── */
.upload-box {
    background: #ffffff;
    border-radius: 14px;
    padding: 24px 28px;
    border: 1px solid #d4e8d8;
    margin-bottom: 20px;
}
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #6a9a7a;
    margin-bottom: 12px;
}

/* ── Result card ── */
.result-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 24px;
    border: 1px solid #d4e8d8;
    margin-top: 16px;
}

/* ── Prediction display ── */
.pred-name {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    font-weight: 400;
    color: #1a3d2b;
    margin: 0 0 4px;
    line-height: 1.1;
}
.pred-cat {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 12px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 20px;
}
.pred-cat.veg { background: #eef6d6; color: #4a7a0a; }
.pred-cat.fruit { background: #fef8e6; color: #a06a0a; }

/* ── Calorie display ── */
.cal-main {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin: 8px 0 6px;
}
.cal-num {
    font-size: 2.8rem;
    font-weight: 600;
    line-height: 1;
}
.cal-unit { font-size: 0.9rem; color: #aaa; }
.cal-tag {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    border-radius: 5px;
    display: inline-block;
    margin-bottom: 14px;
}
.cal-tag.low { background: #eef6d6; color: #4a7a0a; }
.cal-tag.med { background: #fef8e6; color: #a06a0a; }
.cal-tag.high { background: #fdeaea; color: #a02a2a; }

.cal-bar-track {
    height: 4px;
    background: #d4e8d8;
    border-radius: 99px;
    margin: 8px 0 12px;
    overflow: hidden;
}
.cal-bar-fill { height: 100%; border-radius: 99px; }

.portion-line {
    font-size: 0.84rem;
    color: #4a7a5a;
    padding-top: 12px;
    border-top: 1px solid #d4e8d8;
}
.portion-line b { color: #1a3d2b; font-weight: 600; }

/* ── Nutrition grid ── */
.nut-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 16px;
}
.nut-item {
    background: #f2f7f2;
    border-radius: 10px;
    padding: 14px 16px;
    border: 1px solid #d4e8d8;
}
.nut-val {
    font-size: 1.4rem;
    font-weight: 600;
    color: #1a3d2b;
    line-height: 1;
}
.nut-lbl {
    font-size: 0.7rem;
    color: #6a9a7a;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 4px;
}
.nut-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    vertical-align: middle;
}

/* ── Fact & tip ── */
.fact-box {
    background: #f2f7f2;
    border: 1px solid #d4e8d8;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.87rem;
    color: #2a5a3e;
    margin-top: 14px;
    line-height: 1.6;
}
.tip-box {
    border-left: 3px solid #b5d45a;
    background: #f7fae8;
    border-radius: 0 10px 10px 0;
    padding: 13px 18px;
    font-size: 0.87rem;
    color: #4a5a1a;
    margin-top: 12px;
    line-height: 1.6;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 72px 20px;
    color: #aaa;
}
.empty-state .icon {
    font-size: 3.2rem;
    margin-bottom: 16px;
    opacity: 0.5;
}
.empty-state .title {
    font-size: 1rem;
    font-weight: 500;
    color: #888880;
    margin-bottom: 6px;
}
.empty-state .sub {
    font-size: 0.84rem;
    color: #bbb;
}

/* ── Divider ── */
.thin-divider { border: none; border-top: 1px solid #d4e8d8; margin: 14px 0; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #8ab89a;
    font-size: 0.76rem;
    margin-top: 56px;
    padding-bottom: 24px;
    letter-spacing: 0.03em;
}
        /* ── Mobile responsive ── */
@media (max-width: 768px) {
    .hero {
        flex-direction: column;
        padding: 24px 20px;
        gap: 20px;
    }
    .hero h1 {
        font-size: 2rem;
    }
    .hero p {
        font-size: 0.85rem;
    }
    .hero-badge {
        min-width: 80px;
        padding: 12px 16px;
    }
    .stat-row {
        grid-template-columns: 1fr 1fr;
    }
    .nut-grid {
        grid-template-columns: 1fr 1fr;
    }
    .pred-name {
        font-size: 1.8rem;
    }
    .cal-num {
        font-size: 2rem;
    }
}            
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Data & constants ────────────────────────────────
@st.cache_resource
def load_ml_model():
    return load_model('FV.h5')

model = load_ml_model()

labels = {
    0:'apple',1:'banana',2:'beetroot',3:'bell pepper',4:'cabbage',
    5:'capsicum',6:'carrot',7:'cauliflower',8:'chilli pepper',9:'corn',
    10:'cucumber',11:'eggplant',12:'garlic',13:'ginger',14:'grapes',
    15:'jalepeno',16:'kiwi',17:'lemon',18:'lettuce',19:'mango',
    20:'onion',21:'orange',22:'paprika',23:'pear',24:'peas',
    25:'pineapple',26:'pomegranate',27:'potato',28:'raddish',29:'soy beans',
    30:'spinach',31:'sweetcorn',32:'sweetpotato',33:'tomato',34:'turnip',
    35:'watermelon'
}

VEGETABLES = {'Beetroot','Cabbage','Capsicum','Carrot','Cauliflower','Corn',
               'Cucumber','Eggplant','Garlic','Ginger','Lettuce','Onion','Peas',
               'Potato','Raddish','Soy Beans','Spinach','Sweetcorn','Sweetpotato',
               'Tomato','Turnip'}

CALORIE_FALLBACK = {
    'apple':52,'banana':89,'beetroot':43,'bell pepper':31,'cabbage':25,
    'capsicum':31,'carrot':41,'cauliflower':25,'chilli pepper':40,'corn':86,
    'cucumber':16,'eggplant':25,'garlic':149,'ginger':80,'grapes':69,
    'jalepeno':29,'kiwi':61,'lemon':29,'lettuce':15,'mango':60,
    'onion':40,'orange':47,'paprika':282,'pear':57,'peas':81,
    'pineapple':50,'pomegranate':83,'potato':77,'raddish':16,'soy beans':446,
    'spinach':23,'sweetcorn':86,'sweetpotato':86,'tomato':18,'turnip':28,
    'watermelon':31
}

NUTRITION = {
    'apple':(0.3,14,0.2,2.4),'banana':(1.1,23,0.3,2.6),'beetroot':(1.6,10,0.2,2.8),
    'bell pepper':(1.0,6,0.3,2.1),'cabbage':(1.3,6,0.1,2.5),'capsicum':(0.9,6,0.3,2.0),
    'carrot':(0.9,10,0.2,2.8),'cauliflower':(1.9,5,0.3,2.0),'chilli pepper':(1.9,9,0.4,1.5),
    'corn':(3.3,19,1.4,2.7),'cucumber':(0.7,4,0.1,0.5),'eggplant':(1.0,6,0.2,3.0),
    'garlic':(6.4,33,0.5,2.1),'ginger':(1.8,18,0.8,2.0),'grapes':(0.7,18,0.2,0.9),
    'jalepeno':(1.4,7,0.4,2.8),'kiwi':(1.1,15,0.5,3.0),'lemon':(1.1,9,0.3,2.8),
    'lettuce':(1.4,3,0.2,1.3),'mango':(0.8,15,0.4,1.6),'onion':(1.1,9,0.1,1.7),
    'orange':(0.9,12,0.1,2.4),'paprika':(14,54,13,34),'pear':(0.4,15,0.1,3.1),
    'peas':(5.4,14,0.4,5.7),'pineapple':(0.5,13,0.1,1.4),'pomegranate':(1.7,19,1.2,4.0),
    'potato':(2.0,17,0.1,2.2),'raddish':(0.7,4,0.1,1.6),'soy beans':(36,30,20,9.3),
    'spinach':(2.9,4,0.4,2.2),'sweetcorn':(3.3,19,1.4,2.7),'sweetpotato':(1.6,20,0.1,3.0),
    'tomato':(0.9,4,0.2,1.2),'turnip':(0.9,6,0.1,1.8),'watermelon':(0.6,8,0.2,0.4)
}

FUN_FACTS = {
    'apple':"Apples float because 25% of their volume is air.",
    'banana':"Bananas are botanically berries — strawberries aren't.",
    'carrot':"Carrots were originally purple; orange ones were bred in the Netherlands.",
    'tomato':"Tomatoes are legally a vegetable in the US, but botanically a fruit.",
    'watermelon':"Watermelons are 92% water — nature's hydration pack.",
    'spinach':"Spinach loses 90% of its volume when cooked but keeps all nutrients.",
    'lemon':"The acid in lemons can power a small LED bulb.",
    'mango':"Mango trees can live and fruit for over 300 years.",
    'potato':"Potatoes were the first food grown in space (1995).",
    'garlic':"Ancient Egyptians fed garlic to pyramid workers for endurance.",
    'corn':"Corn always has an even number of rows on the cob.",
    'grapes':"Grapes explode when microwaved — do not try at home.",
    'pineapple':"A pineapple takes almost 2 years to grow.",
}

TIPS = {
    'vegetable': "Light steaming preserves more nutrients than boiling.",
    'fruit':     "Eating fresh fruit in the morning maximises vitamin absorption.",
}


# ─────────────────────────── Helpers ────────────────────────────────────────
def fetch_calories(prediction: str):
    query = prediction.lower().strip()
    try:
        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {"search_terms": query, "search_simple": 1, "action": "process",
                  "json": 1, "page_size": 5, "fields": "product_name,nutriments"}
        r = requests.get(url, params=params,
                         headers={"User-Agent": "FoodotoApp/1.0"}, timeout=6)
        r.raise_for_status()
        for p in r.json().get("products", []):
            kcal = p.get("nutriments", {}).get("energy-kcal_100g")
            if kcal is not None:
                return round(float(kcal)), False
    except Exception:
        pass
    if query in CALORIE_FALLBACK:
        return CALORIE_FALLBACK[query], True
    return None, True


def processed_img(img_path: str) -> str:
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img) / 255
    img = np.expand_dims(img, [0])
    y = int(model.predict(img).argmax(axis=-1)[0])
    return labels[y].capitalize()


def cal_color(k):
    if k < 50:   return "#639922"
    if k < 100:  return "#a06a0a"
    return "#a02a2a"

def cal_tag_cls(k):
    if k < 50:   return "low", "Very low calorie"
    if k < 100:  return "low", "Low calorie"
    if k < 200:  return "med", "Moderate calorie"
    return "high", "High calorie"


# ─────────────────────────── Sidebar ─────────────────────────────────────────
with st.sidebar:
    st.image("logo.png", width=80)
    st.markdown("## FOODOTO")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("1. Upload a fruit or vegetable photo\n2. Click **Analyse**\n3. Explore the results")
    st.markdown("---")
    st.markdown("### Settings")
    show_nutrition = st.toggle("Nutrition breakdown", value=True)
    show_chart     = st.toggle("Nutrition chart",      value=True)
    show_fun_facts = st.toggle("Fun facts",            value=True)
    show_tips      = st.toggle("Healthy tips",         value=True)
    grams = st.slider("Portion size (g)", 50, 500, 100, 25)
    st.markdown("---")
    st.markdown("### Supported items")
    t1, t2 = st.tabs(["Vegetables", "Fruits"])
    with t1:
        for v in sorted(VEGETABLES):
            st.markdown(f"- {v}")
    with t2:
        fruits_list = sorted({v.capitalize() for v in labels.values()} - VEGETABLES)
        for f in fruits_list:
            st.markdown(f"- {f}")
    st.markdown("---")
    st.caption("Powered by TensorFlow · Open Food Facts")


# ─────────────────────────── Hero ─────────────────────────────────────────────
st.image("logom.png")
st.markdown("""
<div class="hero">
  <div class="hero-text">
    <h1>Foodoto</h1>
    <p>ML-powered food recognition and calorie counter. Upload a photo for instant nutrition insights.</p>
  </div>
  <div style="display:flex;gap:12px;">
    <div class="hero-badge"><div class="num">36</div><div class="sub">Foods</div></div>
    <div class="hero-badge"><div class="num">Inception V3</div><div class="sub">Powered</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-icon g">🌿</div>
    <div><div class="stat-val">36</div><div class="stat-lbl">Total foods</div></div>
  </div>
  <div class="stat-card">
    <div class="stat-icon b">🥦</div>
    <div><div class="stat-val">20</div><div class="stat-lbl">Vegetables</div></div>
  </div>
  <div class="stat-card">
    <div class="stat-icon o">🍎</div>
    <div><div class="stat-val">16</div><div class="stat-lbl">Fruits</div></div>
  </div>
  <div class="stat-card">
    <div class="stat-icon p">⚡</div>
    <div><div class="stat-val">Live</div><div class="stat-lbl">Calorie data</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────── Upload ──────────────────────────────────────────
st.markdown('<div class="upload-box"><div class="section-label">Upload image</div>', unsafe_allow_html=True)
col_up, col_cam = st.columns([3, 1])
with col_up:
    img_file = st.file_uploader("JPG / PNG", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
with col_cam:
    cam_file = st.camera_input("Camera", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

source = img_file or cam_file


# ─────────────────────────── Results ─────────────────────────────────────────
if source:
    col_img, col_res = st.columns([1, 1], gap="large")

    os.makedirs("./upload_images", exist_ok=True)
    save_path = f"./upload_images/{source.name}"
    with open(save_path, "wb") as fh:
        fh.write(source.getbuffer())

    with col_img:
        st.markdown('<div class="section-label">Your image</div>', unsafe_allow_html=True)
        st.image(Image.open(source), use_container_width=True)

    with col_res:
        st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)
        if st.button("Analyse food", type="primary", use_container_width=True):

            with st.spinner("Running model…"):
                result = processed_img(save_path)

            is_veg   = result in VEGETABLES
            category = "Vegetable" if is_veg else "Fruit"
            cat_cls  = "veg" if is_veg else "fruit"

            st.markdown(f"""
            <div class="pred-name">{result}</div>
            <span class="pred-cat {cat_cls}">{category}</span>
            """, unsafe_allow_html=True)

            # ── Calories ─────────────────────────────────────────────────────
            with st.spinner("Fetching calorie data…"):
                kcal_100, is_est = fetch_calories(result)

            if kcal_100 is not None:
                kcal_portion = round(kcal_100 * grams / 100)
                color = cal_color(kcal_100)
                bar_pct = min(int(kcal_100 / 4), 100)
                tag_cls, tag_label = cal_tag_cls(kcal_100)
                est_note = " · estimated" if is_est else ""

                st.markdown(f"""
                <div class="result-card">
                  <div class="section-label">Calories per 100 g{est_note}</div>
                  <div class="cal-main">
                    <span class="cal-num" style="color:{color}">{kcal_100}</span>
                    <span class="cal-unit">kcal</span>
                  </div>
                  <div class="cal-bar-track">
                    <div class="cal-bar-fill" style="width:{bar_pct}%;background:{color}"></div>
                  </div>
                  <span class="cal-tag {tag_cls}">{tag_label}</span>
                  <hr class="thin-divider">
                  <div class="portion-line">
                    Your <b>{grams} g</b> portion &asymp; <b style="color:{color}">{kcal_portion} kcal</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Could not retrieve calorie data.")

            # ── Nutrition ─────────────────────────────────────────────────────
            if show_nutrition and result.lower() in NUTRITION:
                p, c, fat, fib = NUTRITION[result.lower()]
                f = grams / 100
                st.markdown("&nbsp;")
                st.markdown('<div class="section-label">Nutrition breakdown</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="nut-grid">
                  <div class="nut-item">
                    <div class="nut-val">{round(p*f,1)} g</div>
                    <div class="nut-lbl"><span class="nut-dot" style="background:#4a7abc"></span>Protein</div>
                  </div>
                  <div class="nut-item">
                    <div class="nut-val">{round(c*f,1)} g</div>
                    <div class="nut-lbl"><span class="nut-dot" style="background:#d4a017"></span>Carbohydrates</div>
                  </div>
                  <div class="nut-item">
                    <div class="nut-val">{round(fat*f,1)} g</div>
                    <div class="nut-lbl"><span class="nut-dot" style="background:#c0507a"></span>Fat</div>
                  </div>
                  <div class="nut-item">
                    <div class="nut-val">{round(fib*f,1)} g</div>
                    <div class="nut-lbl"><span class="nut-dot" style="background:#4a9a70"></span>Fibre</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if show_chart:
                    import pandas as pd
                    with st.expander("View nutrition chart"):
                        df = pd.DataFrame({
                            "Nutrient": ["Protein","Carbs","Fat","Fibre"],
                            "Amount (g)": [round(p*f,1), round(c*f,1), round(fat*f,1), round(fib*f,1)]
                        })
                        st.bar_chart(df.set_index("Nutrient"))

            # ── Fun fact ──────────────────────────────────────────────────────
            if show_fun_facts and result.lower() in FUN_FACTS:
                st.markdown(f'<div class="fact-box">💡 {FUN_FACTS[result.lower()]}</div>',
                            unsafe_allow_html=True)

            # ── Tip ───────────────────────────────────────────────────────────
            if show_tips:
                tip_key = "vegetable" if is_veg else "fruit"
                st.markdown(f'<div class="tip-box"><strong>Tip —</strong> {TIPS[tip_key]}</div>',
                            unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">📷</div>
      <div class="title">No image uploaded yet</div>
      <div class="sub">Upload a photo or use your camera to get started</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────── Footer ──────────────────────────────────────────
st.markdown("""
<div class="footer">
  Foodoto &nbsp;·&nbsp; TensorFlow &amp; Streamlit &nbsp;·&nbsp; Nutrition via Open Food Facts
</div>
""", unsafe_allow_html=True)