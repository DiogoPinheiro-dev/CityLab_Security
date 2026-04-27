const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const facesCountEl = document.getElementById("facesCount");
const personsCountEl = document.getElementById("personsCount");
const gesturesCountEl = document.getElementById("gesturesCount");
const latencyValueEl = document.getElementById("latencyValue");
const toggleStreamBtn = document.getElementById("toggleStreamBtn");
const reconnectBtn = document.getElementById("reconnectBtn");
const registrationQr = document.getElementById("registrationQr");
const registrationQrLink = document.getElementById("registrationQrLink");
const registrationUrlEl = document.getElementById("registrationUrl");

const CAPTURE_WIDTH = 960;
const CAPTURE_HEIGHT = 540;
const JPEG_QUALITY = 0.9;
const SERVER_PORT = 8000;
const RECONNECT_BASE_DELAY = 1500;
const RECONNECT_MAX_DELAY = 10000;
const MAX_IN_FLIGHT_FRAMES = 2;

const captureCanvas = document.createElement("canvas");
captureCanvas.width = CAPTURE_WIDTH;
captureCanvas.height = CAPTURE_HEIGHT;
const captureCtx = captureCanvas.getContext("2d");

const state = {
    ws: null,
    reconnectAttempt: 0,
    reconnectTimer: null,
    streamEnabled: true,
    cameraReady: false,
    waitingResponse: false,
    inFlightFrames: 0,
    requestAnimationFrameId: null,
    lastRoundTripMs: null,
    pendingSentAt: null,
    results: {
        rostos: [],
        pessoas: [],
        gestos: [],
    },
};

async function listarCamerasDisponiveis() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        return [];
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((item) => item.kind === "videoinput");
}

function traduzirErroCamera(err, camerasDisponiveis) {
    const quantidade = camerasDisponiveis.length;
    const nome = err && err.name ? err.name : "";

    if (nome === "NotFoundError") {
        if (quantidade === 0) {
            return new Error(
                "Nenhuma camera foi detectada pelo navegador. Conecte/ative uma camera no Windows e tente novamente.",
            );
        }
        return new Error(
            "As cameras foram detectadas, mas nao puderam ser abertas. Feche apps que possam estar usando a camera.",
        );
    }

    if (nome === "NotAllowedError") {
        return new Error("Permissao da camera negada. Libere o acesso para este site no navegador.");
    }

    if (nome === "NotReadableError") {
        return new Error("A camera esta ocupada por outro aplicativo. Feche o app e tente novamente.");
    }

    return err instanceof Error ? err : new Error("Falha desconhecida ao iniciar a camera.");
}

function waitForVideoReadiness(timeoutMs = 8000) {
    return new Promise((resolve, reject) => {
        if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
            resolve();
            return;
        }

        const timeoutId = setTimeout(() => {
            cleanup();
            reject(new Error("A camera nao entregou frames no tempo esperado."));
        }, timeoutMs);

        const onReady = () => {
            if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
                cleanup();
                resolve();
            }
        };

        const cleanup = () => {
            clearTimeout(timeoutId);
            video.removeEventListener("loadedmetadata", onReady);
            video.removeEventListener("loadeddata", onReady);
            video.removeEventListener("canplay", onReady);
        };

        video.addEventListener("loadedmetadata", onReady);
        video.addEventListener("loadeddata", onReady);
        video.addEventListener("canplay", onReady);
    });
}

async function obterStreamComFallback() {
    const tries = [
        {
            video: {
                width: { ideal: CAPTURE_WIDTH },
                height: { ideal: CAPTURE_HEIGHT },
                facingMode: { ideal: "environment" },
            },
            audio: false,
        },
        {
            video: {
                width: { ideal: CAPTURE_WIDTH },
                height: { ideal: CAPTURE_HEIGHT },
            },
            audio: false,
        },
        {
            video: true,
            audio: false,
        },
    ];

    let lastError = null;
    for (const constraints of tries) {
        try {
            return await navigator.mediaDevices.getUserMedia(constraints);
        } catch (err) {
            lastError = err;
        }
    }

    const cameras = await listarCamerasDisponiveis();
    for (const camera of cameras) {
        try {
            return await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: { exact: camera.deviceId },
                    width: { ideal: CAPTURE_WIDTH },
                    height: { ideal: CAPTURE_HEIGHT },
                },
                audio: false,
            });
        } catch (err) {
            lastError = err;
        }
    }

    throw traduzirErroCamera(lastError, cameras);
}

function buildWebSocketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const host = window.location.hostname || "localhost";
    return `${protocol}://${host}:${SERVER_PORT}/stream`;
}

function buildBackendHttpUrl(path) {
    const isHttpPage = window.location.protocol === "http:" || window.location.protocol === "https:";
    if (isHttpPage && window.location.port !== "5500") {
        return path;
    }

    const protocol = isHttpPage ? window.location.protocol : "http:";
    const host = window.location.hostname || "localhost";
    return `${protocol}//${host}:${SERVER_PORT}${path}`;
}

function buildCadastroUrl() {
    const cadastroPath = "/cadastros";
    const isHttpPage = window.location.protocol === "http:" || window.location.protocol === "https:";

    if (isHttpPage && window.location.port !== "5500") {
        return `${window.location.origin}${cadastroPath}`;
    }

    const protocol = isHttpPage ? window.location.protocol : "http:";
    const host = window.location.hostname || "localhost";
    return `${protocol}//${host}:${SERVER_PORT}${cadastroPath}`;
}

async function getCadastroUrl() {
    try {
        const response = await fetch(buildBackendHttpUrl("/access-info"), {
            cache: "no-store",
        });

        if (!response.ok) {
            throw new Error("Falha ao buscar URL publica do cadastro.");
        }

        const payload = await response.json();
        if (payload && typeof payload.cadastro_url === "string") {
            return payload.cadastro_url;
        }
    } catch (err) {
        console.warn("Usando URL local de cadastro como fallback:", err);
    }

    return buildCadastroUrl();
}

async function setupRegistrationQrCode() {
    if (!registrationQr || !registrationQrLink || !registrationUrlEl) {
        return;
    }

    const cadastroUrl = await getCadastroUrl();
    const qrPath = `/qrcode/cadastro.png?url=${encodeURIComponent(cadastroUrl)}`;

    registrationQr.src = buildBackendHttpUrl(qrPath);
    registrationQrLink.href = cadastroUrl;
    registrationUrlEl.textContent = cadastroUrl;
}

function setStatus(message, level) {
    statusEl.textContent = message;
    statusEl.classList.remove("status-ok", "status-warn", "status-error");

    if (level === "ok") {
        statusEl.classList.add("status-ok");
        return;
    }

    if (level === "error") {
        statusEl.classList.add("status-error");
        return;
    }

    statusEl.classList.add("status-warn");
}

function setLatencyDisplay(ms) {
    if (typeof ms !== "number") {
        latencyValueEl.textContent = "-";
        return;
    }

    latencyValueEl.textContent = `${Math.round(ms)} ms`;
}

function updateMetrics() {
    const rostos = Array.isArray(state.results.rostos) ? state.results.rostos : [];
    const pessoas = Array.isArray(state.results.pessoas) ? state.results.pessoas : [];
    const gestos = Array.isArray(state.results.gestos) ? state.results.gestos : [];

    const totalGestosAtivos = gestos.reduce((acc, item) => {
        const alerts = Array.isArray(item.alerts) ? item.alerts.length : 0;
        return acc + (alerts > 0 ? 1 : 0);
    }, 0);

    facesCountEl.textContent = String(rostos.length);
    personsCountEl.textContent = String(pessoas.length);
    gesturesCountEl.textContent = String(totalGestosAtivos);
    setLatencyDisplay(state.lastRoundTripMs);
}

async function iniciarCamera() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("getUserMedia nao suportado nesse navegador/contexto.");
        }

        if (video.srcObject && video.srcObject.getTracks) {
            for (const track of video.srcObject.getTracks()) {
                track.stop();
            }
        }

        const stream = await obterStreamComFallback();

        video.srcObject = stream;
        await video.play().catch(() => null);
        await waitForVideoReadiness();
        state.cameraReady = true;

        setStatus("Camera ativa. Conectando ao servidor...", "warn");
    } catch (err) {
        state.cameraReady = false;
        console.error("Erro ao acessar camera:", err);
        setStatus(err.message || "Falha ao acessar camera. Verifique permissoes.", "error");
        throw err;
    }
}

function closeSocket() {
    if (state.ws) {
        state.ws.onopen = null;
        state.ws.onmessage = null;
        state.ws.onerror = null;
        state.ws.onclose = null;

        if (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING) {
            state.ws.close();
        }
    }

    state.ws = null;
    state.waitingResponse = false;
    state.inFlightFrames = 0;
    state.pendingSentAt = null;
}

function scheduleReconnect() {
    if (state.reconnectTimer) {
        clearTimeout(state.reconnectTimer);
    }

    const delay = Math.min(
        RECONNECT_BASE_DELAY * (state.reconnectAttempt + 1),
        RECONNECT_MAX_DELAY,
    );
    state.reconnectAttempt += 1;

    setStatus(`Sem conexao. Nova tentativa em ${Math.round(delay / 1000)}s...`, "warn");
    state.reconnectTimer = setTimeout(conectarWebSocket, delay);
}

function conectarWebSocket() {
    closeSocket();

    const wsUrl = buildWebSocketUrl();
    const ws = new WebSocket(wsUrl);
    state.ws = ws;

    ws.onopen = () => {
        state.reconnectAttempt = 0;
        if (state.cameraReady) {
            setStatus("Conectado. Processando stream em tempo real.", "ok");
        } else {
            setStatus("Conectado ao servidor, mas sem camera local ativa.", "warn");
        }
    };

    ws.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);
            state.results = {
                rostos: Array.isArray(payload.rostos) ? payload.rostos : [],
                pessoas: Array.isArray(payload.pessoas) ? payload.pessoas : [],
                gestos: Array.isArray(payload.gestos) ? payload.gestos : [],
            };
        } catch (err) {
            console.error("Falha ao interpretar resposta websocket:", err);
        }

        if (typeof state.pendingSentAt === "number") {
            state.lastRoundTripMs = performance.now() - state.pendingSentAt;
        }

        state.pendingSentAt = null;
        state.waitingResponse = false;
        state.inFlightFrames = Math.max(0, state.inFlightFrames - 1);
        updateMetrics();
    };

    ws.onerror = () => {
        setStatus("Erro de conexao com websocket.", "error");
    };

    ws.onclose = () => {
        state.waitingResponse = false;
        state.inFlightFrames = 0;
        if (state.streamEnabled) {
            scheduleReconnect();
        } else {
            setStatus("Conexao encerrada.", "warn");
        }
    };
}

function drawPersons(pessoas) {
    for (const pessoa of pessoas) {
        if (!pessoa || !Array.isArray(pessoa.bbox) || pessoa.bbox.length !== 4) {
            continue;
        }

        const [x1, y1, x2, y2] = pessoa.bbox;
        ctx.strokeStyle = "rgba(14, 165, 233, 0.9)";
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
}

function drawFaces(rostos) {
    for (const rosto of rostos) {
        if (!rosto || !Array.isArray(rosto.bbox) || rosto.bbox.length !== 4) {
            continue;
        }

        const [x1, y1, x2, y2] = rosto.bbox;
        const nome = rosto.nome || "NAO ALUNO";
        const confidence = typeof rosto.confidence === "number" ? Math.round(rosto.confidence * 100) : null;
        const cor = nome === "NAO ALUNO" ? "#dc2626" : "#16a34a";
        const label = confidence === null ? nome : `${nome} (${confidence}%)`;

        ctx.strokeStyle = cor;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.font = "600 14px 'Space Grotesk', sans-serif";
        const textWidth = Math.max(120, ctx.measureText(label).width + 16);
        const textY = Math.max(20, y1 - 8);
        ctx.fillStyle = cor;
        ctx.fillRect(x1, textY - 18, textWidth, 20);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x1 + 8, textY - 3);
    }
}

function drawGestures(gestos) {
    for (const gesto of gestos) {
        if (!gesto || !Array.isArray(gesto.bbox) || gesto.bbox.length !== 4) {
            continue;
        }

        const [x1, y1, x2, y2] = gesto.bbox;
        const alerts = Array.isArray(gesto.alerts) ? gesto.alerts : [];
        const trackId = gesto.track_id ?? "-";
        const hasAlerts = alerts.length > 0;

        ctx.strokeStyle = hasAlerts ? "#dc2626" : "#f59e0b";
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const label = hasAlerts ? `ID ${trackId}: ${alerts.join(" | ")}` : `ID ${trackId}: monitorando`;
        ctx.font = "500 12px 'IBM Plex Mono', monospace";
        const availableWidth = Math.max(80, canvas.width - x1 - 4);
        const textWidth = Math.min(availableWidth, ctx.measureText(label).width + 12);
        const textY = Math.min(canvas.height - 4, Math.max(18, y2 + 16));

        ctx.fillStyle = hasAlerts ? "rgba(220, 38, 38, 0.9)" : "rgba(245, 158, 11, 0.92)";
        ctx.fillRect(x1, textY - 14, textWidth, 16);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x1 + 6, textY - 2, textWidth - 8);
    }
}

function drawFrame() {
    if (video.readyState >= 2 && state.cameraReady) {
        ctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
    } else {
        ctx.fillStyle = "#08131a";
        ctx.fillRect(0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
        ctx.fillStyle = "#9db4c2";
        ctx.font = "600 15px 'Space Grotesk', sans-serif";
        ctx.fillText("Camera local indisponivel", 20, 34);
    }

    drawPersons(state.results.pessoas);
    drawFaces(state.results.rostos);
    drawGestures(state.results.gestos);
}

function sendFrameToBackend() {
    if (!state.streamEnabled || !state.cameraReady) {
        return;
    }

    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
        return;
    }

    if (video.readyState < 2) {
        return;
    }

    if (state.inFlightFrames >= MAX_IN_FLIGHT_FRAMES) {
        return;
    }

    captureCtx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
    state.waitingResponse = true;
    state.pendingSentAt = performance.now();
    state.inFlightFrames += 1;

    captureCanvas.toBlob((blob) => {
        if (!blob) {
            state.waitingResponse = false;
            state.inFlightFrames = Math.max(0, state.inFlightFrames - 1);
            state.pendingSentAt = null;
            return;
        }

        if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
            state.waitingResponse = false;
            state.inFlightFrames = Math.max(0, state.inFlightFrames - 1);
            state.pendingSentAt = null;
            return;
        }

        state.ws.send(blob);
    }, "image/jpeg", JPEG_QUALITY);
}

function frameLoop() {
    drawFrame();
    sendFrameToBackend();
    state.requestAnimationFrameId = requestAnimationFrame(frameLoop);
}

function handleToggleStream() {
    state.streamEnabled = !state.streamEnabled;
    toggleStreamBtn.textContent = state.streamEnabled ? "Pausar Stream" : "Retomar Stream";

    if (state.streamEnabled) {
        setStatus("Retomando stream...", "warn");
        iniciarCamera()
            .then(() => conectarWebSocket())
            .catch(() => null);
    } else {
        setStatus("Stream pausado.", "warn");
        if (state.reconnectTimer) {
            clearTimeout(state.reconnectTimer);
            state.reconnectTimer = null;
        }
        closeSocket();
    }
}

function handleManualReconnect() {
    if (!state.streamEnabled) {
        state.streamEnabled = true;
        toggleStreamBtn.textContent = "Pausar Stream";
    }

    setStatus("Reconectando websocket...", "warn");
    state.reconnectAttempt = 0;
    iniciarCamera()
        .then(() => conectarWebSocket())
        .catch(() => null);
}

function encerrarRecursos() {
    if (state.requestAnimationFrameId) {
        cancelAnimationFrame(state.requestAnimationFrameId);
        state.requestAnimationFrameId = null;
    }

    if (state.reconnectTimer) {
        clearTimeout(state.reconnectTimer);
        state.reconnectTimer = null;
    }

    closeSocket();

    if (video.srcObject && video.srcObject.getTracks) {
        for (const track of video.srcObject.getTracks()) {
            track.stop();
        }
    }
}

async function bootstrap() {
    canvas.width = CAPTURE_WIDTH;
    canvas.height = CAPTURE_HEIGHT;
    updateMetrics();

    toggleStreamBtn.addEventListener("click", handleToggleStream);
    reconnectBtn.addEventListener("click", handleManualReconnect);
    window.addEventListener("beforeunload", encerrarRecursos);
    setupRegistrationQrCode();

    try {
        await iniciarCamera();
        conectarWebSocket();
        frameLoop();
    } catch (err) {
        console.error("Falha na inicializacao:", err);
    }
}

bootstrap();
