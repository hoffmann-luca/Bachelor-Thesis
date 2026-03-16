
# Bachelorarbeit  
## Entwurf und Entwicklung eines KI-gestützten Verfahrens zur Auswahl einer (Meta-)Heuristik für das Tourist-Tour-Design-Problem  
  
Dieses Projekt widmet sich der Entwicklung und Erforschung von Machine-Learning-Methoden zur intelligenten Auswahl einer passenden (Meta-)Heuristik für konkrete Probleminstanzen. Ausgangspunkt ist dabei die Modellierung der Instanz als Graph, was die Grundlage für die Entscheidungsfindung der Lernverfahren bildet. Ziel ist es, ein automatisiertes Modell zu trainieren, das auf Basis der strukturellen Eigenschaften des Eingabegraphen erkennt, welche Heuristik am besten für die jeweilige Problemstellung geeignet ist.  
  
## Reproduktions-Pipeline  
  
Dieser Abschnitt beschreibt, wie die einzelnen Skripte zusammenhängen und wie du die Ergebnisse der Arbeit Schritt für Schritt reproduzieren kannst.  
  
**Wichtig:**  
In vielen Skripten sind absolute Windows-Pfade hart kodiert. 
  
---  
  
## 1. Überblick: Datenfluss  
  
Pipeline:  
  
1. **Instanzen erzeugen oder bereitstellen**  
2. **Heuristiken auf allen Instanzen laufen lassen**  
3. **Graph-Features für alle Instanzen extrahieren**  
4. **Features in Wide-Format + Feature-Subset bringen**  
5. **Heuristik-Ergebnisse + Features mergen**  
6. **Random-Forest-Modelle trainieren**  
7. **Auswertung & Plots**  
8. **Vorhersage für neue Instanzen**  
  
Die wichtigsten Dateien dabei:  
  
- Instanz-Generatoren: `gen_ttdp_instances.ps1`  
- TTDP-Solver & Heuristiken: `core.py`, `Greedy_Solver.py`, `Ils_Solver.py`, `Grasp_Solver.py`, `Vns_Solver.py`, `ttdp_solver.py`  
- Batch-Runs: `parallel_heuristic_runner.py`  
- Feature-Pipeline:  
  `properties_extractor.py`, `batch_PropertiesExtractor.py`,  
  `concat_parquetParts.py`, `transposeFeatureTable.py`, `trimTrainFeatureVector.py`  
- ML-Pipeline: `aggResults_mergeTrainData.py`, `learn_model.py`, `predictOnInstance.py`  
- Auswertung: `expPlots.py`, `secondary_plots.py`, `featurePermutation.py`  
- Visualisierung von Touren: `plotGraph.py`  
  
---  
  
## 2. Instanzen erzeugen / vorbereiten  
  
### 2.1 Künstliche TTDP-Instanzen  
  
**Skript:** `gen_ttdp_instances.ps1`  
  
Erzeugt zufällige TTDP-Instanzen im verwendeten Textformat:  
  
```
n <anzahl_knoten> 
m 1  
tmax <budget>  
x y score    # eine Zeile pro Knoten (erster Knoten = Depot)
 ```
## konkrete Datei-Pipeline

1. `gen_ttdp_instance_auto.py`  
2. `parallel_heuristic_runner.py` -> `result_all.csv`  
3. `batch_PropertiesExtractor.py` → `features-part-*.parquet`  
4. `concat_parquetParts.py` → `features_all.parquet`  
5. `transposeFeatureTable.py` → `features_all_wide.csv`  
6. `trimTrainFeatureVector.py` → `features_all_wide_trim.csv`  
7. `aggResults_mergeTrainData.py` → `train_data.csv`  
8. `learn_model.py` → Modelle (`*.joblib`) + `model_test_results.csv`  
9. `expPlots.py` (und optional `secondary_plots.py`, `featurePermutation.py`)  
10. `predictOnInstance.py` für neue Instanzen

---

# Bachelor's Thesis  
## Design and Development of an AI-Supported Method for Selecting a (Meta-)Heuristic for the Tourist Tour Design Problem  
  
This project is dedicated to the development and investigation of machine-learning methods for the intelligent selection of a suitable (meta-)heuristic for specific problem instances. The starting point is the modeling of each instance as a graph, which forms the basis for the decision-making of the learning methods. The goal is to train an automated model that, based on the structural properties of the input graph, identifies which heuristic is best suited for the respective problem.  
  
## Reproduction Pipeline  
  
This section describes how the individual scripts are connected and how you can reproduce the results of the thesis step by step.  
  
**Important:**  
In many scripts, absolute Windows paths are hard-coded.  
  
---  
  
## 1. Overview: Data Flow  
  
Pipeline:  
  
1. **Generate or provide instances**  
2. **Run heuristics on all instances**  
3. **Extract graph features for all instances**  
4. **Convert features into wide format + feature subset**  
5. **Merge heuristic results + features**  
6. **Train random forest models**  
7. **Evaluation & plots**  
8. **Prediction for new instances**  
  
The most important files involved are:  
  
- Instance generators: `gen_ttdp_instances.ps1`  
- TTDP solver & heuristics: `core.py`, `Greedy_Solver.py`, `Ils_Solver.py`, `Grasp_Solver.py`, `Vns_Solver.py`, `ttdp_solver.py`  
- Batch runs: `parallel_heuristic_runner.py`  
- Feature pipeline:  
  `properties_extractor.py`, `batch_PropertiesExtractor.py`,  
  `concat_parquetParts.py`, `transposeFeatureTable.py`, `trimTrainFeatureVector.py`  
- ML pipeline: `aggResults_mergeTrainData.py`, `learn_model.py`, `predictOnInstance.py`  
- Evaluation: `expPlots.py`, `secondary_plots.py`, `featurePermutation.py`  
- Tour visualization: `plotGraph.py`  
  
---  
  
## 2. Generate / Prepare Instances  
  
### 2.1 Artificial TTDP Instances  
  
**Script:** `gen_ttdp_instances.ps1`  
  
Generates random TTDP instances in the text format used:  
  
```
n <number_of_nodes> 
m 1  
tmax <budget>  
x y score    # one line per node (first node = depot)
 ```
Concrete File Pipeline

## konkrete Datei-Pipeline

1. `gen_ttdp_instance_auto.py`  
2. `parallel_heuristic_runner.py` -> `result_all.csv`  
3. `batch_PropertiesExtractor.py` → `features-part-*.parquet`  
4. `concat_parquetParts.py` → `features_all.parquet`  
5. `transposeFeatureTable.py` → `features_all_wide.csv`  
6. `trimTrainFeatureVector.py` → `features_all_wide_trim.csv`  
7. `aggResults_mergeTrainData.py` → `train_data.csv`  
8. `learn_model.py` → models (`*.joblib`) + `model_test_results.csv`  
9. `expPlots.py` (and optionally `secondary_plots.py`, `featurePermutation.py`)  
10. `predictOnInstance.py` for new Instances
