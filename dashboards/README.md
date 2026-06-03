# Dashboards

In the full platform this folder holds the **Power BI `.pbix`** portfolio ESG heatmap
and a **Streamlit** portfolio explorer.

This prototype is **backend-only** (no frontend, per scope). The same data that would
feed a dashboard is served as JSON from the FastAPI endpoints:

| Dashboard tile        | Backend source                |
|-----------------------|-------------------------------|
| Portfolio ESG heatmap | `GET /portfolio`              |
| Controversy timeline  | `GET /company/{id}` → `risk`  |
| SFDR flag alerts      | `GET /company/{id}` → `sfdr`  |

Point any BI tool at `http://127.0.0.1:8000` to build the visuals.
