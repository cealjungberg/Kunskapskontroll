# =========================
# 1) Importer + custom transformer (måste finnas för att kunna ladda .joblib)
# =========================
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Den här klassen måste finnas i appen eftersom min sparade pipeline innehåller "crop"-steget.
# Annars kan joblib.load krascha (pickle kan inte hitta klassen).
class CenterCrop20x20(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_img = X.reshape(-1, 28, 28)
        X_crop = X_img[:, 4:24, 4:24]
        return X_crop.reshape(-1, 400)


# =========================
# 2) Streamlit + övriga bibliotek
# =========================
import streamlit as st
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# =========================
# 3) Sida / layout (rubrik, tips, struktur)
# =========================
st.set_page_config(page_title="Sifferigenkänning", layout="centered")

st.title("Sifferigenkänning med maskininlärning")
st.caption("Modell tränad på MNIST. Testa att rita en siffra (0–9) eller ladda upp en bild.")

with st.expander("Tips för bättre resultat", expanded=False):
    st.markdown(
        """
- Rita stort och tydligt, gärna i mitten av rutan.  
- Om modellen gissar fel: testa **Invertera färger**.  
- För uppladdade bilder: siffran bör ha bra kontrast mot bakgrunden.
"""
    )

st.divider()


# =========================
# 4) Ladda modellen (cache så den inte laddas om vid varje rerun)
# =========================
@st.cache_resource
def load_model():
    return joblib.load("mnist_model.joblib")

model = load_model()


# =========================
# 5) State för "Testa igen" (tömma canvas + rensa resultat utan att ladda om sidan manuellt)
# =========================
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "result" not in st.session_state:
    st.session_state.result = None  # (pred, top3, preview_arr)

def reset_draw():
    # Ny key gör att canvas blir tom
    st.session_state.canvas_key += 1
    # Rensa sparat resultat
    st.session_state.result = None
    # Kör om appen
    st.rerun()


# =========================
# 6) Hjälpfunktioner (preprocess + prediktion + rendering)
# =========================
def center_digit(img_array: np.ndarray) -> np.ndarray:
    """
    Förbättrad "MNIST-lik" normalisering:
    - Hitta bounding box runt pixlar som > 0
    - Skala så att största sidan blir ~20 pixlar
    - Centrera på en 28x28-canvas
    -  Målet är att förbättra prediktioner på ritade siffror.
    """
    coords = np.column_stack(np.where(img_array > 0))
    if coords.size == 0:
        return img_array

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    digit = img_array[y_min:y_max + 1, x_min:x_max + 1]

    h, w = digit.shape

    # Skala så att största sidan blir cirka 20 pixlar (som MNIST)
    scale = 20 / max(h, w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    digit_img = Image.fromarray(digit)
    digit_img = digit_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    digit = np.array(digit_img, dtype=np.uint8)

    # Lägg den skalade siffran i mitten av en ny 28x28
    new_img = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    new_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit

    return new_img


def preprocess_pil_to_vector(
    pil_img: Image.Image,
    invert: bool,
    threshold_on: bool,
    threshold_value: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess:
    1) Konvertera till gråskala
    2) Resize till 28x28
    3) (Valfritt) invertera
    4) (Valfritt) tröskla för tydligare kontrast
    5) Centrera + skala (MNIST-likt)
    Returnerar:
    - x: (1, 784) för model.predict
    - arr: 28x28 preview för UI
    """
    img = pil_img.convert("L").resize((28, 28), resample=Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.uint8)

    if invert:
        arr = 255 - arr

    if threshold_on:
        arr = np.where(arr > threshold_value, 255, 0).astype(np.uint8)

    # Centrera och skala till MNIST-liknande format
    arr = center_digit(arr)

    x = arr.reshape(1, -1).astype(np.float64)
    return x, arr


def predict_with_confidence(model, x_vec: np.ndarray):
    """
    Prediktera klass + (om möjligt) sannolikheter.
    RandomForest har predict_proba -> vi kan visa topp-3 + % säkerhet.
    """
    pred = int(model.predict(x_vec)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_vec)[0]
    elif hasattr(model, "named_steps") and hasattr(list(model.named_steps.values())[-1], "predict_proba"):
        proba = model.predict_proba(x_vec)[0]
    else:
        return pred, None

    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [(int(i), float(proba[i])) for i in top3_idx]
    return pred, top3


def render_result(pred: int, top3):
    """
    Visa resultatet i appen: prediktion + säkerhet + topp-3.
    """
    st.success(f"**Modellen tror att siffran är: {pred}**")

    if top3 is None:
        st.info("Modellen saknar sannolikhetsestimat (predict_proba), så % säkerhet kan inte visas.")
        return

    best_class, best_p = top3[0]
    st.metric("Säkerhet (topp-gissning)", f"{best_p * 100:.1f} %")

    st.caption("Topp 3 gissningar:")
    for klass, p in top3:
        st.write(f"- **{klass}**: {p * 100:.1f} %")


# =========================
# 7) Sidopanel (inställningar för preprocess + test-knapp)
# =========================
with st.sidebar:
    st.header("Inställningar")
    st.write("Justera vid behov om modellen gissar fel.")

    stroke_width = st.slider("Pennbredd", 5, 40, 18)

    st.subheader("Preprocess")
    invert_draw = st.checkbox("Invertera färger (rita)", value=False)
    invert_up = st.checkbox("Invertera färger (uppladdad)", value=True)

    threshold_on = st.checkbox("Öka kontrast (tröskling)", value=True)
    threshold_value = st.slider("Tröskelvärde", 0, 255, 50)

    st.divider()
    st.button("🔄 Testa igen (töm rutan)", use_container_width=True, on_click=reset_draw)


# =========================
# 8) UI – flikar (rita / ladda upp)
# =========================
tab1, tab2 = st.tabs(["✏️ Rita", "📤 Ladda upp bild"])


# =========================
# 9) Rita-fliken (canvas + prediktion)
# =========================
with tab1:
    st.subheader("Rita en siffra")
    st.caption("Rita tydligt i rutan och klicka sedan på **Prediktera**.")

    col1, col2 = st.columns([1, 1])

    with col1:
        canvas = st_canvas(
            stroke_width=stroke_width,
            stroke_color="white",
            background_color="black",
            width=300,
            height=300,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

        predict_draw = st.button("Prediktera", use_container_width=True)

    with col2:
        st.subheader("Förhandsvisning & resultat")

        if canvas.image_data is not None:
            img = Image.fromarray(canvas.image_data.astype("uint8"), mode="RGBA")

            # Preprocessa för prediktion
            x_vec, preview = preprocess_pil_to_vector(
                img,
                invert=invert_draw,
                threshold_on=threshold_on,
                threshold_value=threshold_value,
            )

            st.image(preview, clamp=True, caption="28×28 efter preprocess (rita)")

            if predict_draw:
                pred, top3 = predict_with_confidence(model, x_vec)
                st.session_state.result = (pred, top3, preview)

        # Visa senast predikterade resultat (så det ligger kvar tills man testar igen)
        if st.session_state.result is not None:
            pred, top3, _ = st.session_state.result
            render_result(pred, top3)


# =========================
# 10) Upload-fliken (ladda upp bild + prediktion)
# =========================
with tab2:
    st.subheader("Ladda upp en bild")
    st.caption("Bilden skalas automatiskt till 28×28 innan prediktion.")

    uploaded = st.file_uploader("Välj en bild (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uppladdad bild", use_container_width=True)

        x_vec, preview = preprocess_pil_to_vector(
            img,
            invert=invert_up,
            threshold_on=threshold_on,
            threshold_value=threshold_value,
        )

        st.image(preview, clamp=True, caption="28×28 efter preprocess (uppladdad)")

        if st.button("Prediktera uppladdad bild", use_container_width=True):
            pred, top3 = predict_with_confidence(model, x_vec)
            render_result(pred, top3)
    else:
        st.info("Ladda upp en bild för att göra en prediktion.")


# =========================
# 11) Möjlig fortsättning/vidareutveckling av appen
# =========================
# Nästa steg hade kunnat vara att spara felexempel och skapa ett nytt dataset för att träna modellen så att den blir bättre på att tolka ritade siffror med olika handstilar

# Om man hade kunnat tala om vilken siffra som var rätt, hade modellen kunnat lära sig av det.

# Man hade kunnat ha en suto-treshold som testar några olika tröskelvärden, kör predict_probs för varje alternativ och väljer den som har högst sannolikhet att stämma

# Normaliseringen kan förbättras genom bl a beskärning (bounding box) och justering av tjocklek/kontrast för att undvika brus

# Vissa andra ML-modeller kan "lära sig av fel" (inkrementellt). RandomForest, som jag använder nu, behöver tränas om med ny data. 

# Däremot kan t ex Logistisk Regression tränas inkrementellt med partial_fit. Dock har den lägre accuracy som utgångspunkt.

# Neurala nät hade kunnat tränas, finjusteras och förbättra prestandan men det hade krävts mer arbete
