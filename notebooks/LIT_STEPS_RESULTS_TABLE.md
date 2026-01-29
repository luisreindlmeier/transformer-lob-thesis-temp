# LiT Verbesserungsschritte – Ergebnisübersicht (0.01 data/test fraction)

Nach jedem Lauf: `results/<ticker>_LiT_<timestamp>/` mit `results.json` und `confusion_matrix_test.png` auswerten.

## Befehle pro Schritt

| Schritt | Beschreibung | Befehl |
|--------|--------------|--------|
| **Initial** | Baseline (ohne Optionen) | `python -m src_prediction run --ticker CSCO --model LiT --epochs 1 --data-fraction 0.01 --test-fraction 0.01` |
| **Step 1** | BiN vor LiT | `... --lit-use-bin` |
| **Step 2** | + event_type Embedding | `... --lit-use-bin --lit-use-event-embed` |
| **Step 3** | + LR-Schedule | `... --lit-use-bin --lit-use-event-embed --lit-use-lr-schedule` |
| **Step 4** | + CLS + Mean-Pool | `... --lit-use-bin --lit-use-event-embed --lit-use-lr-schedule --lit-use-mean-pool` |

Oder alle nacheinander: `bash notebooks/run_lit_steps.sh` (TICKER=CSCO, OUT=./results).

---

## Tabelle: Test Macro F1 und Konfusionsmatrix (Zeilen = True, Spalten = Pred)

*Ergebnisse aus Schnellläufen mit **data-fraction=0.001, test-fraction=0.001** (1 Epoche). Für finale Zahlen mit 0.01/0.01: `bash notebooks/run_lit_steps.sh` ausführen und Werte aus `results/*/results.json` bzw. Konfusionsmatrix-Plot eintragen.*

| Variante | Test Macro F1 | Test Accuracy | F1 Down | F1 Stat | F1 Up | Konfusionsmatrix Test (Kurz) |
|----------|----------------|---------------|---------|---------|-------|------------------------------|
| **Initial** (Baseline, nur EMA) | 0.1502 | 0.1552 | 0.00 | 0.0937 | 0.3570 | Down: 0% korrekt; Stat: ~9%; Up: ~36% (Modell sagt kaum „Down“) |
| **Step 1** (+ BiN) | **0.2279** | 0.5194 | 0.6837 | 0.00 | 0.00 | Down: ~68% korrekt; Stat: 0%; Up: 0% (Modell sagt fast nur „Down“) |
| **Step 2** (+ event_embed) | 0.2279 | 0.5194 | 0.6837 | 0.00 | 0.00 | wie Step 1, keine Änderung |
| **Step 3** (+ LR-Schedule) | 0.2279 | 0.5194 | 0.6837 | 0.00 | 0.00 | wie Step 1, keine Änderung |
| **Step 4** (+ mean_pool) | 0.2279 | 0.5194 | 0.6837 | 0.00 | 0.00 | wie Step 1, keine Änderung |

**Konfusionsmatrix-Kurz:** Zeilenweise (True Label) Recall in % – z. B. „Down: 68%“ = von echten Down 68% als Down vorhergesagt.

---

## Empfehlung (nach Auswertung)

- **Bringt klar etwas:** **BiN (Bilinear Normalization)**. Erhöht Test Macro F1 von 0.15 auf 0.23 und sorgt dafür, dass die Klasse „Down“ überhaupt gelernt wird (statt nur Up/Stat). Ohne BiN kollabiert LiT auf wenige Klassen.
- **Bringt wenig / nichts (bei 0.001 Fraction, 1 Epoche):** **event_type-Embedding, LR-Schedule, Mean-Pool.** In den Schnellläufen keine Verbesserung von Macro F1 oder Konfusionsmatrix; das Modell bleibt nach BiN auf „fast nur Down“ fixiert. Könnte bei mehr Daten/Epochen anders sein.
- **Optional / nur bei Bedarf:**  
  - **LR-Schedule** für längeres Training sinnvoll (Validierungsplateau).  
  - **event_type-Embedding** theoretisch sauber (kategoriales Feature); Effekt erst mit mehr Daten prüfen.  
  - **Mean-Pool** als Alternative zum reinen CLS-Token; hier kein Gewinn sichtbar.

**Fazit:** BiN ist die einzige Maßnahme, die in diesen Läufen einen klaren Gewinn bringt. Danach dominieren vermutlich Datenmenge und Klassenbalance; die weiteren Schritte (Embedding, LR, Pool) lohnen einen erneuten Check mit 0.01 Fraction und mehr Epochen.
