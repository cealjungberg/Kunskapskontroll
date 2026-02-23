# MNIST – Sifferigenkänning med maskininlärning

## 🔗 Live-demo
Appen finns här:  
https://kunskapskontroll-tdq5o2erxvzjpd7xoj8ybm.streamlit.app/

---

Detta projekt innehåller en maskininlärningsmodell som känner igen handskrivna siffror (0–9) baserat på **MNIST-datasetet**.  

En Streamlit-app gör det möjligt att rita en siffra och få modellens prediktion direkt i webbläsaren.

Projektet är genomfört som en **kunskapskontroll i kursen Machine Learning** och visar hela flödet:
- dataanalys  
- modellträning  
- utvärdering  
- enkel applikation  

---

## 📁 Innehåll i repot

- `mnist_app.py` – Streamlit-app  
- `mnist_model.ipynb` – analys, träning och utvärdering  
- Svar på teoretiska frågor (docx)  
- Självutvärdering (docx)  
- `requirements.txt`  
- `mnist_model.joblib` – sparad tränad modell  

⚠️ Modellfilen (`.joblib`) är inte inkluderad i repot på grund av filstorlek.  
Den kan återskapas genom att köra notebooken.

---

## 🚀 Kör appen lokalt

Installera beroenden:

```bash
pip install -r requirements.txt