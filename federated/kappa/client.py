"""
Federated Kappa Client.

Computes additive sufficient statistics (histograms, Σx, XᵀX) per class
and sends them to the server. All kappa computation is performed server-side,
so no raw data ever leaves the client.

Usage (Docker / Flower):
    python federated/kappa/client.py --server_address server:8095 --client_id 1
"""

import argparse
import json
import logging
import warnings

import flwr as fl
import numpy as np
from flwr.common import (
    Code, EvaluateIns, EvaluateRes, FitIns, FitRes,
    GetParametersIns, GetParametersRes, Parameters, Status,
)

from mydataloading import get_client_name, prepare_data

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", default="server:8095")
parser.add_argument("--client_id", type=int, default=1)
args = parser.parse_known_args()[0]

BIN_LO, BIN_HI, N_BINS = -2, 2, 32


# =============================================================================
# Statistics
# =============================================================================

def compute_statistics(X: np.ndarray, y: np.ndarray) -> bytes:
    """Serialise additive sufficient statistics per class to JSON bytes."""
    edges = np.linspace(BIN_LO, BIN_HI, N_BINS + 1)
    nf = X.shape[1]
    classes = {}
    for label in [0, 1]:
        Xc = X[y == label]
        if len(Xc) == 0:
            continue
        classes[str(label)] = {
            "n": len(Xc),
            "histograms": {
                str(j): np.histogram(Xc[:, j], bins=edges)[0].tolist()
                for j in range(nf)
            },
            "feat_sums": np.nansum(Xc, axis=0).tolist(),
            "xtx": (Xc.T @ Xc).tolist(),
        }

    payload = {
        "num_features": nf, "num_bins": N_BINS,
        "bin_lo": BIN_LO, "bin_hi": BIN_HI, "classes": classes,
    }

    def _ser(o):
        if isinstance(o, np.ndarray):   return o.tolist()
        if isinstance(o, np.floating):  return float(o)
        if isinstance(o, np.integer):   return int(o)
        return str(o)

    return json.dumps(payload, default=_ser).encode()


# =============================================================================
# Flower Client
# =============================================================================

class KappaClient(fl.client.Client):
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray):
        self.cid = cid
        self.X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        self.y = np.asarray(y, dtype=np.float64).ravel()
        logger.info(f"Client {cid}: {len(y)} samples, {X.shape[1]} features, "
                    f"class balance={self.y.mean():.3f}")

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(
            status=Status(code=Code.OK, message=""),
            parameters=Parameters(tensor_type="", tensors=[]))

    def fit(self, ins: FitIns) -> FitRes:
        data = compute_statistics(self.X, self.y)
        logger.info(f"Client {self.cid}: sending {len(data) // 1024} KB")
        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=Parameters(tensor_type="", tensors=[data]),
            num_examples=len(self.y),
            metrics={"client_id": self.cid})

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=0.0, num_examples=len(self.y), metrics={})


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    cid = args.client_id
    logger.info(f"Starting kappa client {cid} ({get_client_name(cid)})")

    X_tr, y_tr, X_te, y_te, X_va, y_va, _ = prepare_data(cid)
    X = np.vstack([X_tr.values, X_te.values, X_va.values])
    y = np.concatenate([y_tr, y_te, y_va])

    fl.client.start_client(
        server_address=args.server_address,
        client=KappaClient(cid, X, y))
