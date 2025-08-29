# Quantum Subfield Forecasting Prototype

This repository presents a complete pipeline for analyzing and forecasting the evolution of **39 quantum technology subfields** by combining data from **research publications**, **patent filings**, and **public/private funding sources**.  
The project integrates data collection, cleaning, labeling, modeling, and visualization into one workflow, and includes an interactive **Streamlit app** for exploration.
An analysis on Europe was made at the end and linked to valiate the application forecasted trends. 

---

## Data Sources

### Research Publications
- Collected between **2017–2024**, 25,000 publications.  
- Each publication labeled into one of the **39 verified quantum subfields** with **Claude 3 Haiku API**
- Invalid or uncertain classifications were excluded, so all data is ready for modelling 

### Patents
- **50,000 quantum-related patents** collected from *The Lens* (aligned with 2017–2024).  
- Metadata includes: title, abstract, inventor names, affiliations, and jurisdiction.  
- Labeled into 39 subfields using **Claude 3 Haiku API** (title + abstract) - same labels as for the research dataset.  
- Invalid or uncertain labels removed → final dataset contains **46,106 unique patents**.  
- Help needed to compute a Kappa Score on a subset of 10,000 entires. Please contact us if you want to contribute.  

### Financial Data
- **EU**: Horizon 2020 & Horizon Europe (CORDIS database).  
- **US**: National Quantum Initiative (NQI) reports (2019–2024).  
- **Canada**: National Quantum Strategy (2021–).  
- **Australia**: Sydney Quantum Academy (2019–).  
- **Japan**: Moonshot R&D Program (2020–).  
- **India**: National Quantum Mission (2023–).  
- **China**: National Laboratory for Quantum Information Sciences (2018, ~€14B).  
- **South Korea**: National quantum R&D program (2020–).  
- All values converted to **millions of euros** and split into annual estimates.  

---

## Workflow

1. **Data Collection** → research papers, patents, and funding (2017–2024).  
2. **Preprocessing & Cleaning** → remove duplicates, invalid labels, standardize affiliations/countries and author names 
3. **Labeling** → auto-classification into 39 subfields, manual validation on samples.  
4. **Modeling** → polynomial regression (adaptive degree 1–5) + ridge regularization (α=0.1) to avoid overfitting 
5. **Weighted Predictions** → combine sources with customizable weight settings:  
   - Base (55% patents / 35% research / 10% funding).  
   - Equal weights (33/33/33).  
   - Patents + research only (50/50).  
   - Custom (user-defined).
  
   - This allows the user to study correlation between trends score and any of the 3 data sources used and isolate if desired any source. 

---

## Streamlit Prototype

The **interactive app** allows users to:

- 🔮 Forecast future growth (2025–2028) for any of the 39 subfields.  
- 📊 Compare multiple subfields side by side.  
- 🌍 Explore country level contributions (bar charts + maps). Which countries contributed the most in terms of reseearch/patents/investments within this topics?  
- 📥 Export graphs and data as CSV or PNG.  
- ℹ️ Get explanations of model choices (RMSE, polynomial degree, weights in the interface, good for future testing or reproducing work).  

---

## Purpose

This project helps:

- **Researchers**: track subfield evolution and research focus.  
- **Industry**: identify “hot topics” in patents and innovation.  
- **Policy makers**: understand funding impact and international competition.  

It provides a **transparent, explainable tool** where every modeling step is documented and reproducible.
At the end of the report a real case use of the application is shown and an analysis is made on Europe regarding quantum patents industry, companies founding and the researchers where the market is more biased in working towards to.


---
## Creators:
Alexandru Balan
Irene Colombo




