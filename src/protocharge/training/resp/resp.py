from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence, Dict

import numpy as np

from protocharge.training.linearESPcharges.linear import (
    explicit_solution,
    prepare_linear_system,
)
from protocharge.training.resp_parser import ParseDotXYZ

try:  # SciPy ships the Newton-Krylov solver we target here
    from scipy.optimize import newton_krylov  # type: ignore[attr-defined]
    try:
        from scipy.optimize import NoConvergence  # SciPy >=1.14
    except ImportError:  # pragma: no cover - fall back for older SciPy releases
        from scipy.optimize.nonlin import NoConvergence  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - makes missing dependency explicit at runtime
    newton_krylov = None
    NoConvergence = RuntimeError  # type: ignore[assignment]


@dataclass(frozen=True)
class HyperbolicRestraint:
    """Parameters for the RESP hyperbolic restraint."""

    a: float = 0.0005
    b: float = 0.001
    q0: float = 0.0

    def value(self, charges: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        diff = charges[mask] - self.q0
        term = np.sqrt(diff * diff + self.b * self.b)
        return float(np.sum(self.a * (term - self.b)))

    def gradient(self, charges: np.ndarray, mask: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(charges)
        if not np.any(mask):
            return grad
        diff = charges[mask] - self.q0
        grad_mask = self.a * diff / np.sqrt(diff * diff + self.b * self.b)
        grad[mask] = grad_mask
        return grad


def _resolve_frame_index(frame_index: int | None, total_frames: int) -> int:
    if total_frames == 0:
        raise ValueError("No frames available in geometry file")
    if frame_index is None:
        frame_index = -1
    if frame_index < 0:
        frame_index = total_frames + frame_index
    if frame_index < 0 or frame_index >= total_frames:
        raise IndexError(f"frame_index {frame_index} out of range for {total_frames} frames")
    return frame_index


def _heavy_atom_mask(symbols: Sequence[str]) -> np.ndarray:
    return np.array([sym.upper() != "H" for sym in symbols], dtype=bool)


def _restraint_mask(symbols: Sequence[str], restrain_all_atoms: bool = True) -> np.ndarray:
    """Return a mask selecting atoms that participate in the RESP restraint."""
    if restrain_all_atoms:
        return np.ones(len(symbols), dtype=bool)
    return _heavy_atom_mask(symbols)


def kkt_residual_at(
    q_at_solution: np.ndarray,
    design_matrix: np.ndarray,
    esp_values: np.ndarray,
    symbols: Sequence[str],
    total_charge: float,
    *,
    a: float,
    b: float,
    q0: float = 0.0,
    restrain_all_atoms: bool | None = None,
    restrain_hydrogen: bool | None = None,
) -> Mapping[str, float]:
    """Evaluate first-order KKT residuals for a given RESP charge vector."""

    q = np.asarray(q_at_solution, dtype=float)
    A = np.asarray(design_matrix, dtype=float)
    V = np.asarray(esp_values, dtype=float)

    if q.ndim != 1:
        raise ValueError("q_at_solution must be a 1-D array of charges")

    if A.shape[1] != q.size:
        raise ValueError("Design matrix columns must match number of charges")

    if restrain_all_atoms is None:
        restrain_all_atoms = restrain_hydrogen if restrain_hydrogen is not None else True
    elif restrain_hydrogen is not None:
        raise ValueError("Provide only one of restrain_all_atoms or restrain_hydrogen")

    mask = _restraint_mask(symbols, restrain_all_atoms=restrain_all_atoms)
    restraint = HyperbolicRestraint(a=a, b=b, q0=q0)

    residual = A @ q - V
    gradient = 2.0 * (A.T @ residual) + restraint.gradient(q, mask)

    ones = np.ones_like(q)
    lambda_star = -float(ones @ gradient) / float(ones @ ones)
    kkt_gradient = gradient + lambda_star * ones

    loss_terms = _loss_terms(A, V, q, restraint, mask)

    return {
        "lambda_star": lambda_star,
        "grad_inf_norm": float(np.linalg.norm(kkt_gradient, ord=np.inf)),
        "grad_l2_norm": float(np.linalg.norm(kkt_gradient)),
        "charge_violation": float(q.sum() - total_charge),
        "loss": loss_terms["loss"],
        "ls_term": loss_terms["ls_term"],
        "restraint": loss_terms["restraint"],
    }


def infer_a_from_tc(
    q_at_solution: np.ndarray,
    design_matrix: np.ndarray,
    esp_values: np.ndarray,
    symbols: Sequence[str],
    *,
    b: float,
    q0: float = 0.0,
    restrain_all_atoms: bool | None = None,
    restrain_hydrogen: bool | None = None,
) -> Mapping[str, float]:
    """Infer the restraint strength ``a`` that best fits a given charge vector."""

    q = np.asarray(q_at_solution, dtype=float)
    A = np.asarray(design_matrix, dtype=float)
    V = np.asarray(esp_values, dtype=float)

    if A.shape[1] != q.size:
        raise ValueError("Design matrix columns must match number of charges")

    if restrain_all_atoms is None:
        restrain_all_atoms = restrain_hydrogen if restrain_hydrogen is not None else True
    elif restrain_hydrogen is not None:
        raise ValueError("Provide only one of restrain_all_atoms or restrain_hydrogen")

    mask = _restraint_mask(symbols, restrain_all_atoms=restrain_all_atoms)
    if not np.any(mask):
        raise ValueError("No atoms selected for restraint; cannot infer 'a'")

    residual = A @ q - V
    g = 2.0 * (A.T @ residual)

    h = np.zeros_like(q)
    diff = q[mask] - q0
    h[mask] = diff / np.sqrt(diff * diff + b * b)

    ones = np.ones_like(q)

    hh = float(h @ h)
    h1 = float(h @ ones)
    one_one = float(ones @ ones)
    hg = float(h @ g)
    one_g = float(ones @ g)

    M = np.array([[hh, h1], [h1, one_one]], dtype=float)
    rhs = np.array([-hg, -one_g], dtype=float)

    try:
        a_hat, lambda_hat = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        a_hat, lambda_hat = sol

    grad = g + a_hat * h + lambda_hat * ones

    return {
        "a_hat": float(a_hat),
        "lambda_hat": float(lambda_hat),
        "grad_inf_norm": float(np.linalg.norm(grad, ord=np.inf)),
        "grad_l2_norm": float(np.linalg.norm(grad)),
    }

def load_geometry_symbols(
    geometry_xyz: Path | str,
    *,
    frame_index: int | None = None,
) -> Sequence[str]:
    frames = ParseDotXYZ(geometry_xyz).elements()
    idx = _resolve_frame_index(frame_index, len(frames))
    return frames[idx].symbols


def _loss_terms(
    design_matrix: np.ndarray,
    esp_values: np.ndarray,
    charges: np.ndarray,
    restraint: HyperbolicRestraint,
    mask: np.ndarray,
) -> Mapping[str, float]:
    residual = design_matrix @ charges - esp_values
    ls = float(residual @ residual)
    restraint_value = restraint.value(charges, mask)
    return {
        "ls_term": ls,
        "restraint": restraint_value,
        "loss": ls + restraint_value,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "rrms": float(np.sqrt(np.mean(residual**2) / np.mean(esp_values**2))) if np.any(esp_values) else float("nan"),
    }


def fit_resp_charges(
    resp_out: Path | str,
    esp_xyz: Path | str,
    geometry_xyz: Path | str,
    number_of_atoms: int,
    *,
    frame_index: int | None = None,
    grid_frame_index: int = 0,
    restraint: HyperbolicRestraint | None = None,
    initial_charges: Sequence[float] | None = None,
    total_charge: float | None = None,
    solver_tol: float = 1e-11,
    maxiter: int = 100,
    save_loss_plot: bool = False,
    loss_plot_path: Path | str | None = None,
    show_loss_plot: bool = False,
    restrain_all_atoms: bool = True,
) -> Mapping[str, object]:
    """Run RESP fitting with a hyperbolic restraint via Newton-Krylov."""

    if newton_krylov is None:
        raise ImportError(
            "scipy.optimize.newton_krylov is required for RESP fitting; install scipy to proceed"
        )

    restraint = restraint or HyperbolicRestraint()

    A, V, Q_linear, _esp_charges = prepare_linear_system(
        resp_out,
        esp_xyz,
        number_of_atoms,
        frame_index=frame_index,
        grid_frame_index=grid_frame_index,
    )

    symbols = load_geometry_symbols(geometry_xyz, frame_index=frame_index)
    mask = _restraint_mask(symbols, restrain_all_atoms=restrain_all_atoms)
    if mask.shape[0] != number_of_atoms:
        raise ValueError(
            "Geometry frame atom count does not match requested number_of_atoms"
        )

    if initial_charges is None:
        linear_solution = explicit_solution()
        initial_charges = linear_solution.fit(A, V, Q_linear)["q"]
    q0 = np.asarray(initial_charges, dtype=float)
    if q0.shape != (number_of_atoms,):
        raise ValueError("initial_charges must have length equal to number_of_atoms")

    target_total_charge = float(total_charge if total_charge is not None else Q_linear)

    ones = np.ones_like(q0)
    loss_history: list[float] = []

    def kkt_system(vec: np.ndarray) -> np.ndarray:
        charges = vec[:-1]
        lam = vec[-1]
        residual = A @ charges - V
        grad = 2.0 * (A.T @ residual)
        grad += restraint.gradient(charges, mask)
        grad += lam * ones
        constraint = charges.sum() - target_total_charge
        ls = float(residual @ residual)
        loss_history.append(ls + restraint.value(charges, mask))
        return np.concatenate([grad, np.array([constraint])])

    x0 = np.append(q0, 0.0)

    try:
        solution = newton_krylov(kkt_system, x0, f_tol=solver_tol, maxiter=maxiter)
    except NoConvergence as exc:  # pragma: no cover - surface solver diagnostics clearly
        raise RuntimeError("RESP solver failed to converge") from exc

    charges = solution[:-1]
    lagrange_multiplier = float(solution[-1])

    metrics = dict(_loss_terms(A, V, charges, restraint, mask))
    metrics.update(
        {
            "charges": charges,
            "lagrange_multiplier": lagrange_multiplier,
            "sum_q": float(charges.sum()),
            "target_total_charge": target_total_charge,
            "initial_charges": q0,
            "mask": mask,
            "loss_history": loss_history,
        }
    )
    metrics["predicted_esp"] = A @ charges
    metrics["residual"] = metrics["predicted_esp"] - V

    should_plot = save_loss_plot or show_loss_plot or loss_plot_path is not None
    if should_plot:
        filename: Path | None
        if loss_plot_path is not None:
            filename = Path(loss_plot_path)
        elif save_loss_plot:
            filename = Path(__file__).resolve().parents[2] / "reports" / "resp_loss.png"
        else:
            filename = None

        if filename is not None:
            filename.parent.mkdir(parents=True, exist_ok=True)

        plot_loss_history(metrics["loss_history"], show=show_loss_plot, filename=filename)

    return metrics


def plot_loss_history(
    loss_history: Sequence[float],
    *,
    show: bool = True,
    ax: "matplotlib.axes.Axes" | None = None,
    filename: Path | str | None = None,
) -> "matplotlib.axes.Axes":
    """Plot loss per iteration returned by :func:`fit_resp_charges`.

    Parameters
    ----------
    loss_history
        Iterable of loss values produced during the solve.
    show
        Whether to call ``plt.show()`` before returning. Set ``False`` when
        running inside automated tests or scripts.
    ax
        Optional matplotlib axes to draw on. When ``None`` a new figure/axes is
        created.
    filename
        Optional path. When provided the figure is saved to this location.
    """

    if not loss_history:
        raise ValueError("loss_history is empty; nothing to plot")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "matplotlib is required for plotting RESP loss history"
        ) from exc

    style_path = Path(__file__).resolve().parents[2] / "notebooks" / "prl.mplstyle"
    style_context = plt.style.context(style_path) if style_path.exists() else nullcontext()

    with style_context:
        ax_provided = ax is not None
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        xs = np.arange(len(loss_history))
        (line,) = ax.plot(xs, loss_history, linewidth=1.1, color="C0", label="Loss per iteration")
        ax.scatter(
            xs,
            loss_history,
            s=32,
            facecolor=line.get_color(),
            edgecolor="black",
            linewidth=0.6,
            zorder=line.get_zorder() + 1,
        )
        ax.set_xlabel("Step", fontsize=16)
        ax.set_ylabel(r"$\mathcal{L}$", fontsize=16)
        ax.set_title("RESP loss history")
        ax.grid(True, alpha=0.3)

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight")

        if show:
            plt.show()
        elif not ax_provided:
            plt.close(fig)

    return ax


__all__ = [
    "HyperbolicRestraint",
    "fit_resp_charges",
    "kkt_residual_at",
    "infer_a_from_tc",
    "load_geometry_symbols",
    "plot_loss_history",
]
