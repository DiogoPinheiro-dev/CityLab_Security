"""Ajustes opcionais de CPU, aplicados antes de processar frames."""
import logging
import gc
import os

logger = logging.getLogger(__name__)


def prepare_native_environment(count: int) -> dict[str, str]:
    """Executar antes de importar NumPy, OpenCV, PyTorch ou a API."""
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    if count > 0:
        for name in names:
            os.environ.setdefault(name, str(count))
    return {name: os.environ[name] for name in names if name in os.environ}


def configure_opencv_threads(count: int) -> None:
    if count <= 0:
        return
    import cv2

    cv2.setNumThreads(count)
    logger.info("OpenCV threads: %s", cv2.getNumThreads())


def configure_torch_threads(count: int) -> None:
    if count <= 0:
        return
    import torch

    torch.set_num_threads(count)
    logger.info("PyTorch intra-op threads: %s", torch.get_num_threads())


def ensure_torch_threads(count: int) -> int:
    """Reaplica o limite do PyTorch na thread atual e devolve o valor em uso."""
    import torch

    current = torch.get_num_threads()
    if count > 0 and current != count:
        # O select_device do Ultralytics chama torch.set_num_threads na thread do
        # primeiro predict, e o limite do OpenMP vale por thread: sem reaplicar,
        # cada worker do pipeline roda a pose com um numero diferente de threads.
        torch.set_num_threads(count)
        current = torch.get_num_threads()
    return current


def configure_insight_threads(app, count: int) -> None:
    if count <= 0:
        return
    import onnxruntime as ort

    # InsightFace 0.7 nao encaminha sess_options pelo model_zoo.get_model.
    # Recriar sessoes explicitamente evita uma configuracao silenciosamente ignorada.
    for model in app.models.values():
        previous = model.session
        options = ort.SessionOptions()
        options.intra_op_num_threads = count
        providers = previous.get_providers()
        provider_options = previous.get_provider_options()
        # No Pi de 1 GB, nao manter duas copias da sessao durante a troca.
        # Chamado somente no startup: falha ao recriar deve abortar a carga.
        del model.session
        del previous
        gc.collect()
        model.session = ort.InferenceSession(
            model.model_file,
            sess_options=options,
            providers=providers,
            provider_options=[provider_options.get(provider, {}) for provider in providers],
        )
    logger.info("InsightFace intra-op threads por sessao: %s", count)
