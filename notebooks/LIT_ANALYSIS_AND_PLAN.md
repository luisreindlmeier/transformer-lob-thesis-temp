# Warum LiT schlechter performt als TLOB – Analyse und Maßnahmenplan

## 1. Ausgangslage

- **Gleiche Daten:** Beide Modelle nutzen dieselben Preprocessing-Outputs (`lobster_load`, gleicher Horizon, gleiche Labels 0=Up, 1=Stationary, 2=Down).
- **Gleicher Classification-Threshold:** Labeling mit adaptivem α pro Split ist identisch.
- **Macro F1 als Metrik:** Berücksichtigt Klassenbalance; Ungleichheit Up/Down/Stationary kann die schlechte LiT-Performance **nicht allein** erklären, da TLOB auf denselben Daten gut abschneidet.

**Fazit:** Die Ursache liegt in **Modell-Architektur, Input-Verarbeitung oder Training**, nicht in Daten oder Threshold.

---

## 2. Detaillierter Code-Vergleich

### 2.1 Input-Pipeline (gleiche Datenquelle)

| Aspekt | TLOB | LiT |
|--------|------|-----|
| Daten | `lobster_load(...)` → `(B, 128, 46)` | Identisch |
| Dataset | `LOBDataset(train_x, train_y, SEQ_SIZE)` | Identisch |
| Labels | 0=Up, 1=Stationary, 2=Down | Identisch |

### 2.2 Input-Verarbeitung **vor** dem Transformer

| Aspekt | TLOB | LiT |
|--------|------|-----|
| **Spalte 41 (event_type)** | Categorical: wird als Integer gelesen, mit `nn.Embedding(3, 1)` eingebettet und an die 45 restlichen (kontinuierlichen) Features angehängt. | **Kontinuierlich:** Alle 46 Features gehen in `Linear(46, d_model)`; event_type (Werte 0, 1, 2) wird wie eine reelle Zahl behandelt. |
| **Normalisierung** | **BiN (Bilinear Normalization):** Normalisierung in Zeit- und Feature-Richtung, learnable γ₁, γ₂. Stabilisiert Skalen und Verteilungen. | **Keine** Eingabe-Normalisierung. Raw LOB/Order-Features (sehr unterschiedliche Skalen) gehen direkt in die erste Linear-Schicht. |
| **Positional Encoding** | Sinusoidal (oder learned), additiv auf Embedding. | Learned `pos_emb`, additiv auf Projektion + CLS. |

**Relevanz:**  
- BiN ist in THESIS_MODELS.md als zentrale TLOB-Innovation beschrieben; LiT hat nichts Vergleichbares.  
- event_type als Kategorie (TLOB) vs. als Kontinuum (LiT) kann das Lernverhalten stark beeinflussen.

### 2.3 Architektur

| Aspekt | TLOB | LiT |
|--------|------|-----|
| **Struktur** | Alternating: Attention über Features (dim 46), dann über Zeit (dim 128), mehrfach. Explizite Trennung Feature vs. Zeit. | Standard Transformer: eine Sequenz (128+1 Positionen, d_model 256), nur zeitliche Attention. |
| **Aggregation** | Flatten des letzten verkleinerten Feature×Zeit-Tensors → MLP (352→88→3). | **Nur CLS-Token** (erste Position) → MLP (256→256→3). Gesamte Information muss im CLS ankommen. |
| **Parameter** | Deutlich weniger (hidden_dim 46, 4 Layer alternating). | ~4,85 Mio. (d_model 256, 6 Layer, 8 Heads). |
| **Output-Head** | Mehrstufige MLP mit Dimensionenreduktion. | Zwei Linear (256→256→3) mit GELU/Dropout. |

**Relevanz:**  
- LiT ist schwerer und muss alles in einen CLS-Vektor packen; bei schlechter Eingabe- oder Zwischenrepräsentation kollabiert die Vorhersage leicht.  
- TLOB nutzt keine CLS-Aggregation, sondern einen flachen Feature-Zeit-Vektor.

### 2.4 Training

| Aspekt | TLOB (Lightning) | LiT (train_model) |
|--------|-------------------|-------------------|
| **EMA** | Ja (decay 0.999), Val/Test mit EMA. | Ja (eingebaut), Val mit EMA, Best-Checkpoint = EMA. |
| **Lernrate** | **Adaptiv:** Bei keinem Val-Loss-Improvement → LR halbiert. Bei kleinem Improvement (<0.002) → LR halbiert. | **Konstant** (z. B. 0.0001), keine LR-Anpassung. |
| **Early Stopping** | Patience 1, min_delta 0.002. | Patience 5. |
| **Optimizer** | Adam(lr=cfg.LR, eps=1e-8). | Identisch. |
| **Loss** | CrossEntropyLoss (ohne Class Weights). | Identisch. |

**Relevanz:**  
- TLOB verfeinert die Lösung durch LR-Halbbing; LiT läuft mit fester LR und kann leichter in ein schlechtes Minimum (z. B. „immer Stationary“) laufen oder oszillieren.

---

## 3. Warum das Ergebnis bei LiT schlecht ist – priorisierte Hypothesen

### H1: Keine Eingabe-Normalisierung (höchste Priorität)

- LOB- und Order-Features haben unterschiedliche Skalen und Verteilungen.  
- TLOB normalisiert mit BiN **vor** dem Rest des Netzes.  
- LiT füttert **rohe** Werte in `Linear(46, 256)`. Einige Features können die Aktivierungen und Gradienten dominieren, andere untergehen.  
- **Folge:** Instabiles oder einseitiges Lernen → Tendenz zu einer Klasse (z. B. Stationary) oder schlechter Macro F1.

**Prüfung:** LiT mit einer Eingabe-Normalisierung (z. B. BiN oder LayerNorm/InstanceNorm über Features pro Zeitstep) vor der Projektion testen.

### H2: event_type als kontinuierliches Feature (hohe Priorität)

- In `preprocessing.py` ist event_type kategorial (0, 1, 2 nach Mapping).  
- TLOB: `Embedding(3, 1)` für Spalte 41, Rest kontinuierlich.  
- LiT: Spalte 41 wird wie eine reelle Zahl in `Linear(46, 256)` gezogen.  
- **Folge:** Suboptimale Nutzung von event_type, möglicherweise Verzerrung oder Rauschen in der ersten Schicht.

**Prüfung:** Bei LiT Spalte 41 als Kategorie embedden (wie TLOB) und nur die restlichen 45 kontinuierlichen Features in die Linear-Projektion geben.

### H3: Feste Lernrate + schwieriges Optimum (mittlere Priorität)

- TLOB halbiert die LR bei fehlendem oder kleinem Val-Improvement.  
- LiT nutzt konstante LR; bei großem Modell kann das zu früh eingefrorenen oder kollabierten Lösungen führen.  
- **Folge:** Schlechteres Val/Test-Macro F1 trotz gleicher Daten.

**Prüfung:** Für LiT einen einfachen LR-Schedule (z. B. ReduceLROnPlateau oder manuelles Halbieren wie TLOB) einbauen und gleiche Epochs/Patience wie TLOB verwenden.

### H4: CLS-Aggregation + schlechte Zwischenrepräsentation (mittlere Priorität)

- LiT aggregiert nur über den CLS-Token.  
- Wenn die Eingabe oder die unteren Schichten schlecht skaliert oder verrauscht sind, kann der CLS nur begrenzt alle drei Klassen trennen.  
- **Folge:** Modell „gibt auf“ und driftet in Richtung Mehrheitsklasse.

**Prüfung:** Erst H1/H2 adressieren; wenn LiT dann immer noch schlecht ist, zusätzlich z. B. mean-pool über die Zeit statt nur CLS testen (oder CLS + mean-pool konkatenieren).

### H5: Überfitting / Kapazität (niedrigere Priorität)

- LiT hat deutlich mehr Parameter als TLOB.  
- Bei gleicher Datenmenge könnte LiT eher overfitten oder instabiler trainieren.  
- **Folge:** Gutes Train-Loss, schlechtes Val/Test und schlechter Macro F1.

**Prüfung:** Mehr Dropout, weniger Layer/Heads oder weniger d_model testen, **nachdem** H1–H3 umgesetzt sind.

---

## 4. Maßnahmenplan – wie wir weiter vorgehen

### Phase 1: Input an TLOB angleichen (schneller Gewinn)

1. **BiN vor LiT (oder äquivalent):**  
   - Option A: BiN-Schicht aus TLOB übernehmen; Input `(B, S, F)` wie bei TLOB zu `(B, F, S)` umordnen, BiN anwenden, zurück zu `(B, S, F)` und dann in LiT einspeisen.  
   - Option B: Einfachere Normalisierung (z. B. LayerNorm über die Feature-Dimension pro Zeitstep), um Skalen zu stabilisieren.  
   - **Ziel:** Gleiche Daten, aber stabilere und vergleichbarere Skalen wie bei TLOB.

2. **event_type als Kategorie:**  
   - In LiT: Feature 41 aus dem Tensor herausnehmen, mit `nn.Embedding(3, 1)` (oder kleinerem dim) embedden, mit den anderen 45 Features (kontinuierlich) konkatenieren → z. B. 45+1 = 46 Eingabedimensionen für die bestehende Projektion, oder 45+embed_dim und Projektion anpassen.  
   - **Ziel:** Gleiche semantische Nutzung von event_type wie bei TLOB.

**Erfolgskriterium:** Macro F1 (Val/Test) steigt deutlich und Konfusionsmatrix wird ausgewogener (keine reine Ein-Klassen-Vorhersage).

### Phase 2: Training an TLOB angleichen

3. **Lernrate-Schedule für LiT:**  
   - ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1) oder Logik wie im TLOBTrainer (LR halbieren bei keinem/minimalem Improvement).  
   - **Ziel:** Feineres Optimum, bessere Generalisierung.

4. **Early Stopping:**  
   - Optional: Patience und min_delta an TLOB anpassen (z. B. patience=1, min_delta=0.002), um Vergleichbarkeit zu erhöhen.

**Erfolgskriterium:** Val-Loss und Val-Macro-F1 verbessern sich über mehrere Epochs ohne starkes Overfitting.

### Phase 3: Architektur (nur wenn Phase 1–2 nicht reichen)

5. **CLS vs. Pooling:**  
   - Wenn mit H1/H2/H3 immer noch schlecht: Aggregation erweitern (z. B. mean-pool über Zeit + CLS, dann MLP).  
   - Optional: Kleineres LiT (weniger Layer/d_model) testen, um Overfitting zu prüfen.

6. **Reproduzierbarkeit:**  
   - Feste Seeds, gleiche Epochs und gleiche Daten-Splits für TLOB und LiT; Ergebnisse (Macro F1, Konfusionsmatrix, Val-Loss) protokollieren.

---

## 5. Kurzfassung

| Vermutung | Maßnahme | Priorität |
|-----------|----------|-----------|
| Keine Eingabe-Normalisierung | BiN oder LayerNorm/InstanceNorm vor LiT-Projektion | 1 |
| event_type kontinuierlich | event_type embedden wie bei TLOB | 2 |
| Feste Lernrate | LR-Schedule (Halbieren bei keinem Improvement) | 3 |
| CLS + schlechte Repräsentation | Nach 1–3: Pooling/Verkleinerung testen | 4 |

**Empfehlung:** Zuerst **Phase 1 (BiN + event_type Embedding für LiT)** umsetzen und auf gleichen Daten/Splits mit Macro F1 und Konfusionsmatrix evaluieren. Wenn LiT dann näher an TLOB herankommt, bestätigt das H1/H2; danach Phase 2 (LR-Schedule) ergänzen und bei Bedarf Phase 3 (Aggregation/Kapazität) angehen.
