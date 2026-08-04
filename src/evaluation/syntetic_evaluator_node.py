from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from evaluation.oversmoothing_metrics import compute_oversmoothing_metrics

OVERSMOOTHING_EXCLUDED_MODELS = {
    "linear",
    "linear_lp",
    "linear_hetero_lp",
}


class SynEvaluatorNode:

    def __init__(
        self,
        cfg,
        target_endpoint_names: Optional[List[str]] = None,
        metric_endpoint_names: Optional[List[str]] = None,
    ):
        """
        target_endpoint_names: WSZYSTKIE endpointy, na ktorych model trenuje
            i ktore sa raportowane per-endpoint (eval_per_endpoint=True) -
            bez zmian wzgledem dotychczasowego zachowania.
        metric_endpoint_names: podzbior target_endpoint_names uzywany do
            liczenia AGREGATU (klucze bez prefiksu: "auc", "auprc", ...),
            czyli tego, co steruje selekcja checkpointu (best_metric_name) i
            early stoppingiem. Domyslnie (None) = wszystkie target_endpoint_names,
            zachowanie identyczne jak dotad.

            Uzycie: wyklucz endpointy o bardzo malej liczbie pozytywow
            (np. Serotonin_syndrome, Rhabdomyolysis, Lactic_acidosis - patrz
            analiza wariancji miedzy runami) z AGREGATU, zeby przypadkowy
            wynik na garstce przykladow nie szarpal wyborem "najlepszej"
            epoki - bez usuwania ich z treningu ani z raportowania per-endpoint
            (transparentnosc - w pracy nadal widac ich metryki, tylko nie
            wchodza do glownej, decyzyjnej liczby).
        """
        self.cfg = cfg
        self.target_endpoint_names = target_endpoint_names or []
        self.metric_endpoint_names = (
            list(metric_endpoint_names) if metric_endpoint_names is not None
            else list(self.target_endpoint_names)
        )
        missing = [e for e in self.metric_endpoint_names if e not in self.target_endpoint_names]
        if missing:
            raise ValueError(
                f"metric_endpoint_names {missing} nie sa w target_endpoint_names "
                f"{self.target_endpoint_names} - agregat nie moze zawierac endpointu, "
                "na ktorym model nie trenuje."
            )
        # Pozycje kolumn metric_endpoint_names WEWNATRZ target_endpoint_names -
        # potrzebne do wyciecia podzbioru z y_true/y_prob (ktore sa w kolejnosci
        # target_endpoint_names, patrz get_targeted_labels).
        self._metric_col_idx = [self.target_endpoint_names.index(e) for e in self.metric_endpoint_names]
        self._metric_is_subset = len(self.metric_endpoint_names) < len(self.target_endpoint_names)

        self.per_endpoint = bool(getattr(cfg.training, "eval_per_endpoint", False))

        self.track_oversmoothing = bool(getattr(cfg.training, "track_oversmoothing", True))
        self.oversmoothing_sample_size = int(getattr(cfg.training, "oversmoothing_sample_size", 2000))

        # Metryki oversmoothingu liczone PER WARSTWA (nie tylko na wyjsciu enkodera).
        # Wymaga modelu z encode(data, return_layer_reprs=True) - maja to
        # _PatientDAGGNNBase (sage/gcn/gat/transformer) i RGCNPatientDAGNodeClassifier.
        # Modele bez tego parametru (np. MLP baseline) sa obslugiwane gracefully.
        self.oversmoothing_per_layer = bool(getattr(cfg.training, "oversmoothing_per_layer", False))
        # Domyslnie logujemy tylko agregaty (*_mean + dirichlet), zeby nie zasypac W&B:
        # przy 6 typach wezlow x 4 metryki x (L+1) warstw pelny detal to setki kluczy.
        self.oversmoothing_per_layer_detail = bool(
            getattr(cfg.training, "oversmoothing_per_layer_detail", False)
        )
        # Probkowanie per warstwa moze byc mniejsze - metryki O(N^2) liczone (L+1) razy.
        self.oversmoothing_per_layer_sample_size = int(
            getattr(cfg.training, "oversmoothing_per_layer_sample_size", self.oversmoothing_sample_size)
        )

        model_name = str(getattr(cfg.model, "name", "")).lower()
        excluded = set(getattr(cfg.training, "oversmoothing_excluded_models", OVERSMOOTHING_EXCLUDED_MODELS))
        self.oversmoothing_applicable = model_name not in excluded

        self.classification_threshold = float(getattr(cfg.training, "eval_threshold", 0.5))
        self.auto_threshold = bool(getattr(cfg.training, "auto_threshold", True))
        self._threshold_quantiles = np.linspace(0.02, 0.98, 97)

    # ------------------------------------------------------------------
    # Iteracja po batchach (kluczowa roznica wzgledem starego evaluatora)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_probs(self, model, loader: DataLoader, criterion, device):
        """Iteruje po WSZYSTKICH batchach loadera i agreguje y_true/y_prob,
        zamiast jednego forward() na cala populacje (jak przy jednym duzym
        grafie bipartite). Zwraca rowniez ostatni batch danych (do
        oversmoothing diagnostics) oraz sredni loss."""
        model.eval()
        all_y_true: List[np.ndarray] = []
        all_y_prob: List[np.ndarray] = []
        losses: List[float] = []
        last_batch = None
        n_targets = len(model.target_endpoint_names)

        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            labels = self._get_labels(model, batch)

            logits = logits.view(-1, n_targets)
            labels = labels.view(-1, n_targets)

            loss = criterion(logits, labels)
            losses.append(float(loss.item()))

            probs = torch.sigmoid(logits)
            all_y_true.append(labels.detach().cpu().numpy())
            all_y_prob.append(probs.detach().cpu().numpy())
            last_batch = batch

        y_true = np.concatenate(all_y_true) if all_y_true else np.array([])
        y_prob = np.concatenate(all_y_prob) if all_y_prob else np.array([])
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        return last_batch, y_true, y_prob, mean_loss

    def _get_labels(self, model, batch) -> torch.Tensor:
        """Etykiety w tej samej kolejnosci co logity z model.forward()."""
        from models.TaskB.gnn_node import get_targeted_labels
        return get_targeted_labels(
            batch, model.target_local_idx, target_node_type=model.target_node_type
        )

    def _best_f1_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return self.classification_threshold
        candidates = np.unique(np.quantile(y_prob, self._threshold_quantiles))
        best_f1, best_threshold = -1.0, self.classification_threshold
        for threshold in candidates:
            pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        return best_threshold

    @torch.no_grad()
    def select_threshold(self, model, valid_loader: DataLoader, device) -> float:
        """Osobny, jednorazowy odczyt progu z pelnym forward-passem po
        valid_loader. Zostaje jako publiczne API (np. do jednorazowego uzycia
        poza petla treningowa), ale W PETLI treningowej NIE nalezy jej uzywac
        obok evaluate() - patrz evaluate(..., select_threshold=True), ktore
        wyznacza prog z TEGO SAMEGO forward-passu co metryki, bez podwajania
        kosztu obliczeniowego."""
        if not self.auto_threshold:
            return self.classification_threshold
        _, y_true, y_prob, _ = self._forward_probs(
            model, valid_loader, torch.nn.BCEWithLogitsLoss(), device
        )
        return self._best_f1_threshold(y_true, y_prob)

    # ------------------------------------------------------------------
    # Ewaluacja glowna
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        model,
        loader: DataLoader,
        criterion,
        device,
        threshold: Optional[float] = None,
        select_threshold: bool = False,
    ) -> dict:
        """
        threshold: prog uzywany do metryk progowych (f1/precision/recall).
            Ignorowany, jesli select_threshold=True.
        select_threshold: jesli True, prog jest wyznaczany z TYCH SAMYCH
            y_true/y_prob, ktore wlasnie policzono w tym wywolaniu (jeden
            forward-pass), zamiast osobnego wywolania select_threshold()
            (ktore robiloby DRUGI, niezalezny forward-pass po tym samym
            loaderze). Uzywaj select_threshold=True dla loadera walidacyjnego,
            gdy chcesz swiezy prog co epoke bez dodatkowego kosztu obliczeniowego.
            Wyznaczony prog jest dostepny w zwroconym slowniku pod kluczem
            "classification_threshold" - wyciagnij go stamtad, jesli chcesz
            uzyc tego samego progu do oceny train/test w tej samej epoce.
        """
        last_batch, y_true, y_prob, loss = self._forward_probs(model, loader, criterion, device)

        # Wytnij podzbior kolumn dla progu i agregatu, jesli metric_endpoint_names
        # jest wlasciwym podzbiorem target_endpoint_names (patrz __init__) - np.
        # zeby wykluczyc endpointy z garstka pozytywow (Serotonin_syndrome itp.)
        # z liczby STERUJACEJ selekcja checkpointu, bez usuwania ich z treningu
        # ani z pelnego raportowania per-endpoint ponizej.
        if self._metric_is_subset and y_true.ndim == 2 and y_true.shape[1] == len(self.target_endpoint_names):
            y_true_agg = y_true[:, self._metric_col_idx]
            y_prob_agg = y_prob[:, self._metric_col_idx]
        else:
            y_true_agg, y_prob_agg = y_true, y_prob

        if select_threshold:
            active_threshold = self._best_f1_threshold(y_true_agg, y_prob_agg) if self.auto_threshold else self.classification_threshold
        else:
            active_threshold = self.classification_threshold if threshold is None else float(threshold)

        metrics = self._classification_metrics(y_true_agg, y_prob_agg, prefix="", threshold=active_threshold)
        metrics["loss"] = float(loss)
        metrics["classification_threshold"] = active_threshold

        # Rozbicie per-endpoint: y_true/y_prob sa JUZ w ksztalcie [N, n_targets]
        # (patrz _forward_probs: labels.view(-1, n_targets) per batch, potem
        # concatenate wzdluz osi 0) - reshape ponizej byl wiec bez efektu,
        # ale warunek go strzegacy byl bledny: len(y_true) dla tablicy 2D
        # zwraca TYLKO pierwszy wymiar (liczbe pacjentow), wiec
        # "len(y_true) % n_targets == 0" sprawdzalo w praktyce
        # "liczba_pacjentow_w_loaderze % n_targets == 0" - czysto przypadkowa
        # zaleznosc od konkretnej wielkosci splitu, nie od poprawnosci ksztaltu
        # (ktory jest zawsze poprawny). Stad dzialalo z n_targets=3 (czesciej
        # przypadkowo podzielne) i nie dzialalo z domyslnymi 10.
        if self.per_endpoint and self.target_endpoint_names:
            n_targets = len(self.target_endpoint_names)
            if y_true.ndim == 2 and y_true.shape[1] == n_targets and y_true.shape[0] > 0:
                y_true_reshaped = y_true
                y_prob_reshaped = y_prob
                for i, name in enumerate(self.target_endpoint_names):
                    yt = y_true_reshaped[:, i]
                    yp = y_prob_reshaped[:, i]
                    metrics.update(
                        self._classification_metrics(yt, yp, prefix=f"{name}_", threshold=active_threshold)
                    )

        if (
            self.track_oversmoothing
            and self.oversmoothing_applicable
            and hasattr(model, "encode")
            and last_batch is not None
        ):
            metrics.update(self._oversmoothing_metrics(model, last_batch))

        return metrics

    @torch.no_grad()
    def _oversmoothing_metrics(self, model, batch) -> dict:
        """Metryki oversmoothingu na wyjsciu enkodera, a opcjonalnie takze
        po KAZDEJ warstwie (oversmoothing_per_layer=True).

        Warstwowe reprezentacje pochodza z encode(data, return_layer_reprs=True):
        layer_reprs[0] to wejscie (po input_proj + node_embedding, przed
        pierwsza konwolucja), layer_reprs[i] to wyjscie po i-tej warstwie.
        Dlugosc listy = num_layers + 1.

        Roznica wzgledem dotychczasowego pomiaru: metryki "plaskie"
        (oversmoothing/...) pokazuja, jak reprezentacja KONCOWA zmienia sie w
        czasie treningu. Metryki warstwowe (oversmoothing/layer{i}/...) pokazuja,
        jak reprezentacja degraduje sie W GLAB SIECI w pojedynczym forward-passie
        - i to jest wlasciwe pytanie o oversmoothing.
        """
        out: dict = {}

        if not self.oversmoothing_per_layer:
            z_dict = model.encode(batch)
            out.update(
                compute_oversmoothing_metrics(z_dict, batch, sample_size=self.oversmoothing_sample_size)
            )
            return out

        try:
            z_dict, layer_reprs = model.encode(batch, return_layer_reprs=True)
        except TypeError:
            # Model bez obslugi return_layer_reprs (np. MLP baseline) - cichy
            # fallback do pomiaru tylko na wyjsciu, zamiast wywalac trening.
            z_dict = model.encode(batch)
            out.update(
                compute_oversmoothing_metrics(z_dict, batch, sample_size=self.oversmoothing_sample_size)
            )
            return out

        # Pomiar koncowy - te same klucze co dotychczas, zeby nie zerwac
        # istniejacych wykresow/porownan w W&B.
        out.update(
            compute_oversmoothing_metrics(z_dict, batch, sample_size=self.oversmoothing_sample_size)
        )

        for layer_idx, layer_z_dict in enumerate(layer_reprs):
            layer_metrics = compute_oversmoothing_metrics(
                layer_z_dict, batch, sample_size=self.oversmoothing_per_layer_sample_size
            )
            for key, value in layer_metrics.items():
                short = key[len("oversmoothing/"):] if key.startswith("oversmoothing/") else key
                if not self.oversmoothing_per_layer_detail:
                    # Tylko agregaty: *_mean oraz globalne warianty energii Dirichleta.
                    keep = short.endswith("_mean") or short in {
                        "dirichlet_energy",
                        "dirichlet_energy_relation_macro",
                        "dirichlet_energy_edge_weighted",
                    }
                    if not keep:
                        continue
                out[f"oversmoothing/layer{layer_idx}/{short}"] = value

        return out

    def _classification_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray, prefix: str, threshold: float
    ) -> dict:
        # y_true.size (nie len(y_true)!) - dla wywolania per-endpoint (1D)
        # oba daja to samo, ale dla wywolania agregatowego (prefix="", y_true
        # 2D [n_pacjentow, n_targets]) len() zwraca TYLKO pierwszy wymiar
        # (liczbe pacjentow), podczas gdy .sum() sumuje WSZYSTKIE elementy -
        # bez tej poprawki auprc_baseline byl ~n_targets-krotnie zawyzony,
        # a auprc_lift proporcjonalnie zanizony. Dzieki temu, ze kazdy
        # endpoint ma tu zawsze te sama liczbe wierszy (ci sami pacjenci na
        # wszystkich endpointach), n = y_true.size daje wynik matematycznie
        # rownowazny macro-usrednieniu baseline po endpointach - spojny z
        # tym, jak roc_auc_score/average_precision_score juz licza AUC/AUPRC
        # (macro dla danych 2D).
        n = int(y_true.size)
        positives = int(y_true.sum()) if n > 0 else 0
        negatives = n - positives

        out = {
            f"{prefix}auc": float("nan"),
            f"{prefix}auprc": float("nan"),
            f"{prefix}brier": float("nan"),
            f"{prefix}f1": float("nan"),
            f"{prefix}precision": float("nan"),
            f"{prefix}recall": float("nan"),
            f"{prefix}positives": positives,
            f"{prefix}negatives": negatives,
            f"{prefix}auprc_baseline": float(positives / n) if n > 0 else float("nan"),
            f"{prefix}auprc_lift": float("nan"),
        }

        if positives == 0 or negatives == 0:
            return out

        out[f"{prefix}auc"] = float(roc_auc_score(y_true, y_prob))
        out[f"{prefix}auprc"] = float(average_precision_score(y_true, y_prob))
        out[f"{prefix}brier"] = float(np.mean((y_prob - y_true) ** 2))

        baseline = out[f"{prefix}auprc_baseline"]
        if baseline > 0:
            out[f"{prefix}auprc_lift"] = float(out[f"{prefix}auprc"] / baseline)

        y_pred = (y_prob >= threshold).astype(int)
        out[f"{prefix}f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        out[f"{prefix}precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        out[f"{prefix}recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

        return out